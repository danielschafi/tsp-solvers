import argparse
from pathlib import Path

import h5py
import numpy as np
import tsplib95

from src.data_handling.tsplib_extension import TSPProblemWithOSMIDs


def _load_tsp_file(tsp_file: Path | str) -> TSPProblemWithOSMIDs:
    """
    Loads a .tsp file with the TSPProblemWithOSMIDs format
    """
    if not Path(tsp_file).exists():
        raise FileNotFoundError(f"tsp_file: {tsp_file} does not exist.")

    tsp_file = Path(tsp_file)
    problem = tsplib95.load(tsp_file, problem_class=TSPProblemWithOSMIDs)
    return problem


def _extract_relevant_parts_from_tsp_problem(
    problem: TSPProblemWithOSMIDs,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Extracts the relevant parts from the tsp problem and converts them to the right format
    """
    adj = np.array(problem.edge_weights)
    dim = problem.dimension
    coords = np.array(problem.node_locations)
    return adj, coords, dim


def _full_to_upper_tri(full_adj: np.ndarray) -> np.ndarray:
    """
    Converts the full adjacency list to the upper triangle, not saving the diagonal (offset k=1)
    Since our problems are symmetric
    """
    return full_adj[np.triu_indices(full_adj.shape[0], k=1)].astype(np.uint16)


def _process_tsp_dir(problems_dir: Path):
    """Save all the tsp files in the directory into a single hdf5 container file"""
    files = sorted(Path(problems_dir).glob("*.tsp"))
    num_problems = len(files)
    output_file = Path(problems_dir) / "processed.h5"

    if output_file.exists():
        print(
            f"Already exists: {output_file}, skipping. Remove or rename it if you want to run this again"
        )
        return

    print(f"Preprocessing {num_problems} files in {problems_dir}")

    with h5py.File(output_file, "w") as f:
        for idx, tsp_file in enumerate(files):
            progress = idx / num_problems

            # Just for fun :D
            print(
                f"Processing .tsp file {idx}/{num_problems}:\t ["
                + "=" * int(progress * 50)
                + " " * int((1 - progress) * 50)
                + "]"
            )
            # Get Data from tsplib file
            problem = _load_tsp_file(tsp_file)
            adj, coords, dim = _extract_relevant_parts_from_tsp_problem(problem)
            upper_tri = _full_to_upper_tri(adj)

            # Save them in hdf5. Groups named as padded integers 00000, 00001 etc.
            grp = f.create_group(f"{idx:05d}")
            grp.create_dataset(
                "adj_upper_tri",
                data=upper_tri,
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("coords", data=coords)
            grp.attrs["name"] = tsp_file.stem

        f.attrs["num_problems"] = num_problems
        f.attrs["dim"] = dim

    print(f"  -> {output_file}  ({output_file.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Convert the .tsp files from a directory into an object that is easily loadable into a pytorch dataset for training a GNN."
    )
    arg_parser.add_argument(
        "--src_dir",
        type=str,
        default="data/gnn_data",
        help="Directory containing tsp files of one size. Or the parent directory containing the directories of samples one dir per size. data/gnn_data/{10,20,100,..} or data/gnn_data",
    )
    arg_parser.add_argument(
        "--out_dir",
        type=str,
        default="data/tsp_dataset",
        help="Directory to save the dataset.",
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = arg_parser.parse_args()
    src_dir, out_dir, seed = (
        args.src_dir,
        args.out_dir,
        args.seed,
    )

    if len(list(Path(src_dir).glob("*.tsp"))) > 0:
        # only one dir
        _process_tsp_dir(src_dir)
    else:
        # containing multiple problem sizes
        for obj in Path(src_dir).iterdir():
            if Path(obj).is_dir() and len(list(Path(obj).glob("*.tsp"))) > 0:
                # todo check if already exists the hdf5 file
                _process_tsp_dir(obj)


if __name__ == "__main__":
    main()
