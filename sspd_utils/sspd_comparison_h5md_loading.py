#!/usr/bin/env python
"""Kicked MDMin relaxations (trainer_it10_0 MACE + D3) of the last frame of each
traj/sspd*xyz, kicked forward (+2x forces) and reversed (-2x forces)."""
import h5py
import numpy as np
from ase import Atoms
from ase.io import write
from pathlib import Path
import sys

import glob

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.optimize import MDMin
from mace.calculators import MACECalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

D3_CUTOFF = 40.0  # Bohr


class FirstStepOverride(Calculator):
    """Adds user-supplied forces for the first 10 calculations,
    then falls back to the wrapped calculator for everything after."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, real_calc, custom_forces):
        super().__init__()
        self.real_calc = real_calc
        self.custom_forces = np.array(custom_forces)
        self.step_done = 0

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Always get real energy/forces so we have something sane for energy
        self.real_calc.calculate(atoms, properties=['energy', 'forces'],
                                 system_changes=system_changes)
        real_energy = self.real_calc.results['energy']
        real_forces = self.real_calc.results['forces']

        if self.step_done < 10:
            self.results['forces'] = real_forces + self.custom_forces.copy()
            self.results['energy'] = real_energy  # keep energy physical
            self.step_done += 1
        else:
            self.results['forces'] = real_forces
            self.results['energy'] = real_energy

def load_last_h5md(path: str) -> Atoms:
    with h5py.File(path, "r") as f:
        positions = f["data/positions"][-1]
        numbers = f["data/atomic_numbers"][-1]
        cell = f["data/cell"][-1]
        pbc = f["data/pbc"][-1].astype(bool)
        step = int(f["steps/positions"][-1])

    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
    atoms.info["step"] = step
    return atoms

def read_sspd_last():
    snapshots = []
    for i in range(50):
        tmp_path = traj_folder/f"sspd_{i}.h5md"
        tmp_atoms = load_last_h5md(tmp_path)
        snapshots.append(tmp_atoms)
    return snapshots

def load_last_h5md_weights(path: str) -> Atoms: 
    with h5py.File(path, "r") as f:
        atomic_numbers = f["data/atomic_numbers"][0]
        pbc = f["data/pbc"][:].astype(bool)
        n_atoms = atomic_numbers.shape[0]

        last = f["data/positions"].shape[0] - 1
        positions = f["data/positions"][last]
        cell = f["data/cell"][last]

        last_w = f["data/weight_vector"].shape[0] - 1
        weights = f["data/weight_vector"][last_w].reshape(n_atoms, 3)
    # print(weights)
    weights = weights / np.linalg.norm(weights, axis=1).max()

    atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
    atoms.arrays["forces"] = weights#*10
    return atoms

def read_sspd_weights_last(traj_folder,n_ens=50):
    snapshots = []
    for i in range(n_ens):
        tmp_path = traj_folder/f"sspd_{i}.h5md"
        tmp_atoms = load_last_h5md_weights(tmp_path)
        snapshots.append(tmp_atoms)
    return snapshots

def load_h5md(path: str) -> list[Atoms]:
    with h5py.File(path, "r") as f:
        atomic_numbers = f["data/atomic_numbers"][0]          # (
        positions_all  = f["data/positions"][:]               # (n_pos, n,
        cell_all       = f["data/cell"][:]                    # (n_pos, 3,
        pbc            = f["data/pbc"][:].astype(bool)        # (
        pos_steps      = f["steps/positions"][:]              # (n_
        n_atoms = atomic_numbers.shape[0]
        weight_vector_all = None
        weight_map = None
        if "weight_vector" in f["data"]:
            weight_vector_all = f["data/weight_vector"][:]        # (n_w, 1, n*3)
            weight_vector_all = weight_vector_all.reshape(
                weight_vector_all.shape[0], n_atoms, 3
            )
            weight_steps = f["steps/weight_vector"][:]            # (n_w,)
            weight_map = dict(zip(weight_steps.tolist(), weight_vector_all))

    frames = []
    for i, step in enumerate(pos_steps):
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions_all[i],
            cell=cell_all[i],
            pbc=pbc,
        )
        if weight_map is not None:
            weights = weight_map.get(int(step))
            if weights is not None:
                norm = np.linalg.norm(weights, axis=1).max()
                if norm > 0:
                    weights = weights / norm
                atoms.arrays["forces"] = weights
        frames.append(atoms)

    return frames

def tmp_compare():
    traj_mace_sspd_customint =Path("/tmp/testOct30/seed12_Thigh2000_Tbase500_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1_custom/run_Cu1Omin/traj/")
    traj_mace_sspd_tsint = Path("/tmp/testOct30/seed12_Thigh2000_Tbase500_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1_ts/run_Cu1Omin/traj/")
    sspd_mace_sspd_customint = read_sspd_weights_last(traj_mace_sspd_customint,20)
    sspd_mace_sspd_tsint = read_sspd_weights_last(traj_mace_sspd_tsint,20)
    write("sspd_mace_sspd_customint.xyz",sspd_mace_sspd_customint)
    write("sspd_mace_sspd_tsint.xyz",sspd_mace_sspd_tsint)
    return sspd

def test_weights_loading():
    Thigh = 2000
    traj_mace_sspd_customint = Path(f"/home/mvondrak/projects/CO2onCu_decomposition/CuCO_final/test_sspd_integrators/mace_sspd_customint/seed12_Thigh{Thigh}_Tbase200_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1/run_Cu1Omin/traj/")
    traj_mace_sspd_customint_subHes = Path(f"/home/mvondrak/projects/CO2onCu_decomposition/CuCO_final/test_sspd_integrators/mace_sspd_customint_subHes/seed12_Thigh{Thigh}_Tbase200_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1/run_Cu1Omin/traj/")
    traj_mace_sspd_tsint = Path(f"/home/mvondrak/projects/CO2onCu_decomposition/CuCO_final/test_sspd_integrators/mace_sspd_tsint/seed12_Thigh{Thigh}_Tbase200_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1/run_Cu1Omin/traj/")
    traj_mace_sspd_tsint_subHes = Path(f"/home/mvondrak/projects/CO2onCu_decomposition/CuCO_final/test_sspd_integrators/mace_sspd_tsint_subHes/seed12_Thigh{Thigh}_Tbase200_sspd500_lang100_dt0.000025_MACE_RPBED3zero_com1/run_Cu1Omin/traj/")

    sspd_mace_sspd_customint = read_sspd_weights_last(traj_mace_sspd_customint,20)
    sspd_mace_sspd_customint_subHes = read_sspd_weights_last(traj_mace_sspd_customint_subHes,20)
    sspd_mace_sspd_tsint = read_sspd_weights_last(traj_mace_sspd_tsint,20)
    sspd_mace_sspd_tsint_subHes = read_sspd_weights_last(traj_mace_sspd_tsint_subHes,20)
    write("sspd_mace_sspd_customint.xyz",sspd_mace_sspd_customint)
    write("sspd_mace_sspd_customint_subHes.xyz",sspd_mace_sspd_customint_subHes)
    write("sspd_mace_sspd_tsint.xyz",sspd_mace_sspd_tsint)
    write("sspd_mace_sspd_tsint_subHes.xyz",sspd_mace_sspd_tsint_subHes)
    return sspd

def main():

    
    traj_folder = Path("/home/mvondrak/projects/CO2onCu_decomposition/CuCO_final/sspds/sspds_claude_musinn/seed12_Thigh2000_Tbase200_sspd5000_lang5000_dt0.000025_MACE_RPBED3zero_com1/run_Cu1Omin/traj")
    mols = read_sspd_last()
    real_calc = MACECalculator(model_path='../MACE_RPBED3zero_com1.model', head=None,
                       default_dtype="float64", device='cuda')
    for sspd_traj,mol in enumerate(mols):
        # print(sspd_traj)
        # reversed kick
        custom_forces = mol.get_forces() * -2
        mol.calc = FirstStepOverride(real_calc, custom_forces)
        opt = MDMin(mol, dt=0.05, trajectory=f"{sspd_traj}_kicked_min_rev.traj")
        opt.run(fmax=0.5)

        # forward kick
        mol = read(f"{sspd_traj}@{-1}", format="extxyz")
        custom_forces = mol.get_forces() * 2
        mol.calc = FirstStepOverride(real_calc, custom_forces)
        opt = MDMin(mol, dt=0.05, trajectory=f"{sspd_traj}_kicked_min.traj")
        opt.run(fmax=0.5)
        
if __name__ == "__main__":
    main()
