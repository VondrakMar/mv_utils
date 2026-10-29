import h5py
import numpy as np
from ase import Atoms
from ase.io import write
import sys

import argparse

def load_h5md(path: str) -> list[Atoms]:
    with h5py.File(path, "r") as f:
        atomic_numbers = f["data/atomic_numbers"][0]          # (N,)
        positions_all  = f["data/positions"][:]               # (n_pos, N, 3)
        cell_all       = f["data/cell"][:]                    # (n_pos, 3, 3)
        pbc            = f["data/pbc"][:].astype(bool)        # (3,)
        # energy_all     = f["data/potential_energy"][:, 0]     # (n_e,)
        # temp_all       = f["data/temperature"][:, 0]          # (n_e,)
        pos_steps      = f["steps/positions"][:]              # (n_pos,)
        # print(positions_all)
        '''
        energy_steps   = f["steps/potential_energy"][:]       # (n_e,)
        ke_all         = f["data/kinetic_energy"][:, 0]       # (n_e,)
    energy_map = dict(zip(energy_steps, energy_all))
    ke_map     = dict(zip(energy_steps, ke_all))
    temp_map   = dict(zip(energy_steps, temp_all))
    '''
        n_atoms = atomic_numbers.shape[0]

        weight_vector_all = None
        weight_map = None
        if "weight_vector" in f["data"]:
            # saved as (n_steps, 1, n_atoms * 3) -> reshape per-frame to (n_atoms, 3)
            weight_vector_all = f["data/weight_vector"][:]        # (n_w, 1, N*3)
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
        # atoms.info["energy"]         = float(energy_all[i])
        # atoms.info["kinetic_energy"] = float(ke_all[i])
        # atoms.info["temperature"]    = float(temp_all[i])
        if weight_map is not None:
            weights = weight_map.get(int(step))
            if weights is not None:
                norm = np.linalg.norm(weights, axis=1).max()
                if norm > 0:
                    weights = weights / norm
                # store as "forces" so ase gui / extxyz round-trip it as a
                # per-atom vector field and lets you view it with the
                # built-in force-vector visualization
                atoms.arrays["forces"] = weights
        frames.append(atoms)

    return frames

def convert_h5_xyz(h5md_path: str, out_path: str | None = None) -> str:
    frames = load_h5md(h5md_path)
    write(out_path, frames, format="extxyz")
    return out_path

def convert_h5_traj(h5md_path: str, out_path: str | None = None) -> str:
    frames = load_h5md(h5md_path)
    write(out_path, frames)#, format="extxyz")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",help="Where is the h5 file",type=str)
    parser.add_argument("--output_file",help="Name of the xyz file",type=str)
    args = parser.parse_args()
    convert_h5_traj(h5md_path=args.input_file,out_path=args.output_file)
