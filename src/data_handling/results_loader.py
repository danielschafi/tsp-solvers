"""
Loads per-problem benchmark result JSON files into a single pandas DataFrame.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# The columns that we care about for visualization.
CORE_COLUMNS = [
    "solver",
    "problem",
    "problem_size",
    "cost",
    "time_to_solve",
    "solution_status",
    "valid_solution",
    "timed_out_without_tour",
    "timestamp",
    "additional_metadata",
]


def load_results(results_dir: Path = Path("results")) -> pd.DataFrame:
    """
    Loads all problem JSON result files under results_dir into a DataFrame.

    The additional_metadata column is kept as a dict. To flatten it for a specific
    solver use pd.json_normalize(df[df.solver == "gurobi"]["additional_metadata"]).

    drops duplicates
    """
    results_dir = Path(results_dir)
    rows = []

    for json_file in results_dir.glob("**/*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {json_file}: {e}")
            continue

        row = {col: data.get(col) for col in CORE_COLUMNS}

        # Ensure additional_metadata is always a dict
        if not isinstance(row["additional_metadata"], dict):
            row["additional_metadata"] = {}

        rows.append(row)

    if not rows:
        logger.warning(f"No result JSON files found under {results_dir}")
        return pd.DataFrame({col: pd.Series(dtype=object) for col in CORE_COLUMNS})

    df = pd.DataFrame(rows)

    # sorts and then drops duplicate rows by (solver, problem) keeping newest
    df = (
        df.sort_values("timestamp", na_position="first")
        .drop_duplicates(subset=["solver", "problem"], keep="last")
        .sort_values(["solver", "problem_size", "problem"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Loaded {len(df)} results from {results_dir} "
        f"({df['solver'].nunique()} solvers, {df['problem_size'].nunique()} problem sizes)"
    )
    return df


if __name__ == "__main__":
    results_dir = Path("results")
    df = load_results(results_dir)
    print(df.shape)
    print(df.groupby(["solver", "problem_size"]).size().to_string())
