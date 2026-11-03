#!/usr/bin/env python
"""Kicked MDMin relaxations (trainer_it10_0 MACE + D3) of the last frame of each
traj/sspd*xyz, kicked forward (+2x forces) and reversed (-2x forces)."""
import torch

from ase.io import read, write
from ase.optimize import MDMin
from ase.visualize import view

from pathlib import Path
from mace.calculators import MACECalculator


from torch_sim.trajectory import TorchSimTrajectory

from first_step_override import FirstStepOverride
from sspd_process import read_sspd_last, read_sspd_weights_last,load_last_h5md

dtype = torch.float64
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def save_trajectory_xyz(traj_path, out_path=None):
    """Read one .h5md trajectory and save it as .xyz (same name, same folder by default)."""
    traj_path = Path(traj_path)
    if out_path is None:
        out_path = traj_path.with_suffix(".xyz")   # replica_001.h5md -> replica_001.xyz

    atoms_list = read_trajectory(traj_path, state_return=False)
    write(out_path, atoms_list, format="extxyz")
    return out_path

def save_all_trajectories_xyz(base_path, n_members=50, md_prefix="replica", padded_zeros=1):
    """Save every replica trajectory as .xyz into base_path."""
    base_path = Path(base_path)
    width = len(str(n_members)) + padded_zeros

    saved = []
    for member in range(0, n_members):
        traj = base_path / f"{md_prefix}_{member:0{width}d}.h5md"
        saved.append(save_trajectory_xyz(traj))
    return saved

def read_trajectory(traj_path, device=device, dtype=dtype, state_return=False):
    """Read a single trajectory file; return atoms (and optionally states)."""
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    ts_traj = TorchSimTrajectory(traj_path, mode="r")
    n_frames = len(ts_traj)
    atoms_list = [ts_traj.get_atoms(frame=i) for i in range(n_frames)]
    states = None
    if state_return:
        states = [ts_traj.get_state(frame=i, device=device, dtype=dtype)
                  for i in range(n_frames)]
    ts_traj.close()

    return (atoms_list, states) if state_return else atoms_list

def get_trajectories(base_path, n_members=50, md_prefix="replica", padded_zeros=1,
                     device=device, dtype=dtype, state_return=False):
    """Loop over all replicas and collect their trajectories."""
    base_path = Path(base_path)


    width = len(str(n_members)) + padded_zeros

    all_atoms, all_states = [], []
    for member in range(0, n_members):
        traj = base_path / f"{md_prefix}_{member:0{width}d}.h5md"
        result = read_trajectory(traj, device=device, dtype=dtype, state_return=state_return)
        if state_return:
            atoms_list, states = result
            all_states.append(states)
        else:
            atoms_list = result
        all_atoms.append(atoms_list)
    return (all_atoms, all_states) if state_return else all_atoms

    
def read_nvt(T=300):
    folder_name = "Cu1O_runsDiffTemps_overdamped_jobcraft_folder_20261103_141840"
    # folder_name = "Cu1O_runsDiffTemps_jobcraft_folder_20261103_130057"
    traj = Path(f'/tmp/{folder_name}/MD_NVT_T{T}/nvt_results/')
    # traj = Path(f'/tmp/run1_NVT_jobcraft_folder_20261103_121133/nvt_results/')
    list_ase= get_trajectories(traj,50,"replica",padded_zeros=1)
    return list_ase



    
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
    mols = read_sspd_last(traj_folder)
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
