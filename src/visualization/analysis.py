"""
Aggregate statistics behind the benchmark plots in
`src/visualization/viz_benchmark_results.py`.

Produces per-(solver, problem_size) summaries for:
  * tour cost            -> `cost_stats`
  * time to solve        -> `time_stats`
  * optimality gap (%)   -> `gap_stats`

Plus solver coverage (counts and valid-solution rate) and headline numbers
useful for captioning the plots.

Usage:
    uv run -m src.visualization.analysis
    uv run -m src.visualization.analysis --solvers MCTSOnly UTSPSolver
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_handling.results_loader import load_results

logger = logging.getLogger(__name__)


AGG_FUNCS = ["count", "mean", "std", "min", "median", "max"]


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["valid_solution"] == True].copy()  # noqa: E712


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """How many instances we have per (solver, problem_size), and how many are valid."""
    grp = df.groupby(["solver", "problem_size"], dropna=False)
    cov = grp.agg(
        n_runs=("problem", "count"),
        n_valid=("valid_solution", lambda s: int(s.fillna(False).sum())),
        n_timed_out=(
            "timed_out_without_tour",
            lambda s: int(s.fillna(False).sum()),
        ),
    )
    cov["valid_rate"] = cov["n_valid"] / cov["n_runs"]
    return cov.reset_index()


def cost_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Tour-cost summary per (solver, problem_size). Valid solutions only."""
    data = _valid_rows(df)
    if data.empty:
        return pd.DataFrame()
    out = (
        data.groupby(["solver", "problem_size"])["cost"]
        .agg(AGG_FUNCS)
        .reset_index()
    )
    return out


def time_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Solve-time summary per (solver, problem_size). Positive times only."""
    data = df[df["time_to_solve"].notna() & (df["time_to_solve"] > 0)].copy()
    if data.empty:
        return pd.DataFrame()
    out = (
        data.groupby(["solver", "problem_size"])["time_to_solve"]
        .agg(AGG_FUNCS)
        .reset_index()
    )
    return out


def _reference_table(
    df: pd.DataFrame, reference_solver: str
) -> tuple[pd.DataFrame | None, str | None]:
    """Pick the first available reference solver and return its (problem, cost) table."""
    data = _valid_rows(df)
    for ref in [reference_solver, "gurobi", "concorde"]:
        cand = data[data["solver"] == ref][["problem", "cost"]].rename(
            columns={"cost": "ref_cost"}
        )
        if not cand.empty:
            return cand, ref
    return None, None


def gap_stats(
    df: pd.DataFrame, reference_solver: str = "concorde"
) -> tuple[pd.DataFrame, str | None]:
    """
    Optimality-gap (%) summary per (solver, problem_size), relative to the
    reference solver. Returns (stats_df, used_reference_name).
    """
    ref_df, used_ref = _reference_table(df, reference_solver)
    if ref_df is None:
        logger.warning("No reference solver results found; skipping gap stats.")
        return pd.DataFrame(), None

    other = _valid_rows(df)
    other = other[other["solver"] != used_ref]
    merged = other.merge(ref_df, on="problem", how="inner")
    if merged.empty:
        logger.warning("No overlap with reference solver; skipping gap stats.")
        return pd.DataFrame(), used_ref

    merged["optimality_gap_pct"] = (
        (merged["cost"] - merged["ref_cost"]) / merged["ref_cost"] * 100
    )
    out = (
        merged.groupby(["solver", "problem_size"])["optimality_gap_pct"]
        .agg(AGG_FUNCS)
        .reset_index()
    )
    return out, used_ref


def pairwise_cost_diff(df: pd.DataFrame, solver_a: str, solver_b: str) -> pd.DataFrame:
    """
    Per-instance head-to-head comparison: cost(solver_a) - cost(solver_b),
    aggregated per problem_size. Useful for "is A consistently better than B".
    """
    data = _valid_rows(df)
    a = data[data["solver"] == solver_a][["problem", "problem_size", "cost"]].rename(
        columns={"cost": f"cost_{solver_a}"}
    )
    b = data[data["solver"] == solver_b][["problem", "cost"]].rename(
        columns={"cost": f"cost_{solver_b}"}
    )
    if a.empty or b.empty:
        logger.warning(
            f"pairwise_cost_diff: missing data for {solver_a} or {solver_b}."
        )
        return pd.DataFrame()

    merged = a.merge(b, on="problem", how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["abs_diff"] = merged[f"cost_{solver_a}"] - merged[f"cost_{solver_b}"]
    merged["rel_diff_pct"] = (
        merged["abs_diff"] / merged[f"cost_{solver_b}"] * 100
    )
    merged[f"{solver_a}_wins"] = merged["abs_diff"] < 0
    merged[f"{solver_b}_wins"] = merged["abs_diff"] > 0

    out = (
        merged.groupby("problem_size")
        .agg(
            n_instances=("problem", "count"),
            mean_abs_diff=("abs_diff", "mean"),
            mean_rel_diff_pct=("rel_diff_pct", "mean"),
            median_rel_diff_pct=("rel_diff_pct", "median"),
            **{
                f"{solver_a}_win_rate": (f"{solver_a}_wins", "mean"),
                f"{solver_b}_win_rate": (f"{solver_b}_wins", "mean"),
            },
        )
        .reset_index()
    )
    return out


def headline_numbers(df: pd.DataFrame) -> dict:
    """One-line takeaways useful for plot captions."""
    out: dict = {}
    valid = _valid_rows(df)
    if not valid.empty:
        out["solvers"] = sorted(valid["solver"].unique().tolist())
        out["problem_sizes"] = sorted(valid["problem_size"].unique().tolist())
        out["instances_per_size"] = (
            valid.groupby("problem_size")["problem"].nunique().to_dict()
        )

    timing = df[df["time_to_solve"].notna() & (df["time_to_solve"] > 0)]
    if not timing.empty:
        slowest = (
            timing.groupby(["solver", "problem_size"])["time_to_solve"]
            .mean()
            .sort_values(ascending=False)
            .head(3)
        )
        fastest = (
            timing.groupby(["solver", "problem_size"])["time_to_solve"]
            .mean()
            .sort_values(ascending=True)
            .head(3)
        )
        out["slowest_mean_times"] = slowest.round(3).to_dict()
        out["fastest_mean_times"] = fastest.round(3).to_dict()
    return out


def _print_section(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(no data)")
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda x: f"{x:,.4f}",
    ):
        print(df.to_string(index=False))


def run(
    results_dir: Path = Path("results"),
    reference_solver: str = "concorde",
    solvers: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    df = load_results(results_dir)
    if df.empty:
        logger.warning("No results found.")
        return {}

    if solvers is not None:
        df = df[df["solver"].isin(solvers)].copy()

    cov = coverage(df)
    cost = cost_stats(df)
    time_df = time_stats(df)
    gap, used_ref = gap_stats(df, reference_solver=reference_solver)

    _print_section("Coverage (runs per solver x size)", cov)
    _print_section("Tour cost (valid solutions only)", cost)
    _print_section("Solve time (seconds)", time_df)
    _print_section(
        f"Optimality gap vs {used_ref} (%)" if used_ref else "Optimality gap",
        gap,
    )

    # Common head-to-head if both solvers are present.
    solver_set = set(df["solver"].unique())
    if {"MCTSOnly", "UTSPSolver"}.issubset(solver_set):
        pair = pairwise_cost_diff(df, "MCTSOnly", "UTSPSolver")
        _print_section("Head-to-head: MCTSOnly vs UTSPSolver (per instance)", pair)
    else:
        pair = pd.DataFrame()

    head = headline_numbers(df)
    print("\n=== Headline numbers ===")
    for k, v in head.items():
        print(f"{k}: {v}")

    results = {
        "coverage": cov,
        "cost_stats": cost,
        "time_stats": time_df,
        "gap_stats": gap,
        "pairwise_mcts_vs_utsp": pair,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in results.items():
            if frame is None or frame.empty:
                continue
            path = out_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            logger.info(f"Wrote {path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark statistics.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--reference_solver",
        type=str,
        default="concorde",
        help="Reference solver for optimality gap (fallback: gurobi, then concorde).",
    )
    parser.add_argument(
        "--solvers",
        nargs="*",
        default=None,
        help="Restrict analysis to these solvers (default: all).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="If set, write CSVs of each summary table here.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.set_printoptions(suppress=True)
    run(
        results_dir=args.results_dir,
        reference_solver=args.reference_solver,
        solvers=args.solvers,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
