from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class TSPDataset(Dataset):
    def __init__(self, problems_h5_container: Path = Path("data/gnn_data/200")) -> None:
        super().__init__()
        self.file_path = Path(problems_h5_container)
        self._file = None

        with h5py.File(self.file_path, "r") as f:
            self.num_problems = int(f.attrs["num_problems"])
            self.dim = int(f.attrs["dim"])

    def __len__(self) -> int:
        return self.num_problems

    def _get_file(self) -> h5py.File:
        """Load the h5 file if not loaded and return it"""
        if self._file is None:
            self._file = h5py.File(self.file_path, "r", swmr=True)
        return self._file

    def __getitem__(self, idx):
        # load one of the tsp problems from the h5 container
        grp = self._get_file()[f"{idx:05d}"]

        # adj matrix is stored as upper tri, reconstruct full adj matrix
        upper_tri = torch.tensor(grp["adj_upper_tri"], dtype=torch.float32)
        adj = torch.zeros((self.dim, self.dim))
        tri_idx = torch.triu_indices(self.dim, self.dim, offset=1)
        adj[tri_idx[0], tri_idx[1]] = upper_tri
        adj = adj + adj.T

        coords = torch.tensor(grp["coords"], dtype=torch.float32)
        return {"adj": adj, "coords": coords}

    def __del__(self):
        if self._file is not None:
            self._file.close()


def main():
    dataset = TSPDataset(Path("data/gnn_data/200_test/processed.h5"))
    print(f"Number of problems in dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Adjacency matrix shape: {sample['adj'].shape}")
    print(f"Coordinates shape: {sample['coords'].shape}")
    print(sample["adj"])


if __name__ == "__main__":
    main()
