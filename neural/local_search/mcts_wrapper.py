"""Python wrapper around the C++ MCTS TSP solver in Search/.

Prerequisites
-------------
- The `test` binary must be compiled in Search/:
    cd Search && make
- The binary's Rec_Num (Search/code/include/TSP_IO.h) must equal topk_idx.shape[2].
- Max_Inst_Num in TSP_IO.h must be >= the number of instances you pass (default 128).
"""

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_MAX_INST = 128  # must match Max_Inst_Num in TSP_IO.h
_N_THREADS = 32  # parallelism used by the shell scripts


def _write_input_file(
    path: Path,
    dist_matrix: np.ndarray,  # [N, n, n]  integer travel times
    topk_idx: np.ndarray,  # [N, n, k]  0-based
    topk_val: np.ndarray,  # [N, n, k]
) -> None:
    N, n, k = topk_idx.shape
    with open(path, "w") as f:
        for i in range(N):
            # n×n travel-time matrix, row-major
            dist_str = " ".join(
                str(int(dist_matrix[i, r, c])) for r in range(n) for c in range(n)
            )
            # dummy opt tour: 1 2 ... n, closed with 1 (n+1 ints total)
            tour_str = " ".join(str(j + 1) for j in range(n)) + " 1"
            # top-k indices, 1-based, flat over all nodes
            idx_str = " ".join(
                str(int(topk_idx[i, j, l]) + 1) for j in range(n) for l in range(k)
            )
            # top-k values, flat over all nodes
            val_str = " ".join(
                f"{float(topk_val[i, j, l]):.6f}" for j in range(n) for l in range(k)
            )
            f.write(
                f"{dist_str} output {tour_str} indices {idx_str} output {val_str}\n"
            )


def _parse_result_file(path: Path) -> dict[int, list[int]]:
    """Return {0-based inst_index -> 0-based tour} from a statistics result file."""
    tours: dict[int, list[int]] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Inst_Index:"):
            inst_idx = int(lines[i].split("Inst_Index:")[1].split()[0]) - 1
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("Solution:"):
                i += 1
            if i < len(lines):
                cities = [int(x) - 1 for x in lines[i].replace("Solution:", "").split()]
                tours[inst_idx] = cities
        i += 1
    return tours


def run_mcts(
    dist_matrix: np.ndarray,
    topk_idx: np.ndarray,
    topk_val: np.ndarray,
    search_dir: str | Path = "bin/MCTS-UTSP",
    n_threads: int = _N_THREADS,
    use_rec: bool = True,
    rec_only: bool = False,
    max_candidate_num: int = 5,
    max_depth: int = 10,
    alpha: float = 1.0,
    beta: float = 10.0,
    param_h: float = 3.0,
    restart: bool = False,
) -> list[list[int]]:
    """Run the C++ MCTS solver on a batch of TSP instances.

    Args:
        dist_matrix:      [N, n, n] integer travel-time matrix. Diagonal is ignored
                          (the binary sets Distance[i][i] = Inf_Cost internally).
        topk_idx:         [N, n, k] top-k neighbor indices per node (0-based).
                          k must equal Rec_Num compiled into the binary.
        topk_val:         [N, n, k] GNN heat-map scores for those neighbors.
        search_dir:       Path to Search/ (must contain the compiled 'test' binary).
        n_threads:        Parallel solver processes (splits the batch across threads).
        use_rec:          Pass GNN heat map to MCTS as edge priors.
        rec_only:         Restrict the candidate set to GNN top-k edges only.
        max_candidate_num: Size of the 2-opt / MCTS candidate neighbourhood.
        max_depth:        Max action depth in MCTS.
        alpha:            UCB exploration coefficient.
        beta:             Back-propagation update rate.
        param_h:          Sampling multiplier (controls simulation budget per step).
        restart:          Restart from random solution when no improvement is found.

    Returns:
        List of N tours, each a list of 0-based city indices.
    """
    search_dir = Path(search_dir).resolve()
    binary = search_dir / "test"
    if not binary.exists():
        raise FileNotFoundError(
            f"MCTS binary not found at {binary}. Run 'make' inside {search_dir}."
        )

    N, n, _ = topk_idx.shape
    if N > _MAX_INST:
        raise ValueError(
            f"run_mcts handles at most {_MAX_INST} instances per call "
            f"(Max_Inst_Num in TSP_IO.h), got {N}."
        )

    # Pad to exactly _MAX_INST so the binary reads the right number of entries.
    pad = _MAX_INST - N
    if pad > 0:
        dist_in = np.concatenate([dist_matrix, np.tile(dist_matrix[-1:], (pad, 1, 1))])
        idx_in = np.concatenate([topk_idx, np.tile(topk_idx[-1:], (pad, 1, 1))])
        val_in = np.concatenate([topk_val, np.tile(topk_val[-1:], (pad, 1, 1))])
    else:
        dist_in, idx_in, val_in = dist_matrix, topk_idx, topk_val

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        input_file = tmpdir / "instances.txt"
        _write_input_file(input_file, dist_in, idx_in, val_in)

        def _run_batch(batch_idx: int) -> tuple[int, Path]:
            result_file = tmpdir / f"result_{batch_idx}.txt"
            cmd = [
                str(binary),
                str(batch_idx),
                str(result_file),
                str(input_file),
                str(n),
                str(int(use_rec)),
                str(int(rec_only)),
                str(max_candidate_num),
                str(max_depth),
                str(alpha),
                str(beta),
                str(param_h),
                str(int(restart)),
                "0",  # restart_reconly
            ]
            subprocess.run(
                cmd,
                cwd=search_dir,
                stdin=subprocess.DEVNULL,  # unblocks getchar() at end of main()
                capture_output=True,
                check=True,
            )
            return batch_idx, result_file

        tours_by_idx: dict[int, list[int]] = {}
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(_run_batch, i) for i in range(n_threads)]
            for fut in as_completed(futures):
                _, result_file = fut.result()
                tours_by_idx.update(_parse_result_file(result_file))

    return [tours_by_idx[i] for i in range(N)]
