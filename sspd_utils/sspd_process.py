"""Loading helpers for SSPD h5md trajectories."""
import h5py
import numpy as np
from ase import Atoms

# def read_h5md_trajectory(traj_folder, n_ens=50):
#     snapshots = []
#     for i in range(n_ens):
#         tmp_path = traj_folder/f"sspd_{i}.h5md"
#         tmp_atoms = load_last_h5md(tmp_path)
#         snapshots.append(tmp_atoms)
#     return snapshots


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

def read_sspd_last(traj_folder, n_ens=50):
    snapshots = []
    for i in range(n_ens):
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
