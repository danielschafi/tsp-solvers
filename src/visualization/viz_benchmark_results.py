"""
Aggregate benchmark plots across solvers and problem sizes.

Usage:
    uv run -m src.visualization.viz_benchmark --results_dir results --out_dir results/plots
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_handling.results_loader import load_results

logger = logging.getLogger(__name__)


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a valid tour and a real cost."""
    return df[df["valid_solution"] == True].copy()  # type: ignore[return-value]  # noqa: E712


def plot_cost_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean tour cost per solver across problem sizes (log-log)."""
    data = _valid_rows(df)
    if data.empty:
        logger.warning("No valid solutions to plot for cost_vs_size.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=data,
        x="problem_size",
        y="cost",
        hue="solver",
        errorbar="sd",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Problem size (n nodes)")
    ax.set_ylabel("Tour cost")
    ax.set_title("Tour cost vs problem size")
    ax.legend(title="Solver")
    fig.tight_layout()
    out_path = out_dir / "cost_vs_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_time_vs_size(df: pd.DataFrame, out_dir: Path) -> None:
    """Mean solve time per solver across problem sizes (log-log)."""
    data = df[df["time_to_solve"].notna() & (df["time_to_solve"] > 0)].copy()
    if data.empty:
        logger.warning("No timing data to plot for time_vs_size.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=data,
        x="problem_size",
        y="time_to_solve",
        hue="solver",
        errorbar="sd",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Problem size (n nodes)")
    ax.set_ylabel("Time to solve (s)")
    ax.set_title("Solve time vs problem size")
    ax.legend(title="Solver")
    fig.tight_layout()
    out_path = out_dir / "time_vs_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_optimality_gap(
    df: pd.DataFrame, out_dir: Path, reference_solver: str = "concorde"
) -> None:
    """
    Optimality gap (%) relative to a reference solver, per problem instance.

    gap = (solver_cost - ref_cost) / ref_cost * 100

    Only computed for (problem_size, problem) pairs where both the solver
    and the reference have a valid solution.
    """
    data = _valid_rows(df)

    # Try reference_solver, fall back to gurobi
    ref_solvers = [reference_solver, "gurobi"]
    ref_df = None
    used_ref = None
    for ref in ref_solvers:
        candidate = data[data["solver"] == ref][["problem", "cost"]].rename(  # type: ignore[index]
            columns={"cost": "ref_cost"}
        )
        if not candidate.empty:
            ref_df = candidate
            used_ref = ref
            break

    if ref_df is None:
        logger.warning(
            "No reference solver results found; skipping optimality gap plot."
        )
        return

    # Join and compute gap
    other = data[data["solver"] != used_ref].copy()
    merged = other.merge(ref_df, on="problem", how="inner")
    if merged.empty:
        logger.warning(
            "No overlapping problems between solvers and reference; skipping gap plot."
        )
        return

    merged["optimality_gap_pct"] = (
        (merged["cost"] - merged["ref_cost"]) / merged["ref_cost"] * 100
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(
        data=merged,
        x="problem_size",
        y="optimality_gap_pct",
        hue="solver",
        errorbar="sd",
        marker="o",
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Problem size (n nodes)")
    ax.set_ylabel(f"Optimality gap vs {used_ref} (%)")
    ax.set_title(f"Optimality gap vs {used_ref}")
    ax.legend(title="Solver")
    fig.tight_layout()
    out_path = out_dir / "optimality_gap_vs_size.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_all(
    results_dir: Path,
    out_dir: Path,
    reference_solver: str = "concorde",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        logger.warning("No results found — nothing to plot.")
        return

    plot_cost_vs_size(df, out_dir)
    plot_time_vs_size(df, out_dir)
    plot_optimality_gap(df, out_dir, reference_solver=reference_solver)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot aggregate benchmark results.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/plots"))
    parser.add_argument(
        "--reference_solver",
        type=str,
        default="concorde",
        help="Solver used as optimality reference (default: concorde, fallback: gurobi).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plot_all(args.results_dir, args.out_dir, reference_solver=args.reference_solver)


if __name__ == "__main__":
    main()
