#!/usr/bin/env python
"""Kicked MDMin relaxations (trainer_it10_0 MACE + D3) of the last frame of each
traj/sspd*xyz, kicked forward (+2x forces) and reversed (-2x forces)."""

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


def main():
    real_calc = SumCalculator([
        MACECalculator(model_path='trainer_it10_0.model', head=None,
                       default_dtype="float64", device='cuda'),
        TorchDFTD3Calculator(device='cuda', damping='bj', xc='rpbe',
                             dtype=torch.float64, cutoff=D3_CUTOFF * units.Bohr),
    ])

    sample_dir = sorted(glob.glob("traj/sspd*xyz"))
    print(sample_dir)

    for sspd_traj in sample_dir:
        print(sspd_traj)
        # reversed kick
        mol = read(f"{sspd_traj}@{-1}", format="extxyz")
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
