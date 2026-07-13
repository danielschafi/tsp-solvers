"""UTSP paper appendix Table tb:sp search parameters (NeurIPS 2023).

Paper T (expand actions) maps to Param_H via Param_H * n = T.
Param_T (wall-clock seconds multiplier) is from the UTSP README, not the table.
n=25 uses the TSP-20 row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchParams:
    alpha: float
    beta: float
    max_candidate_num: int  # M
    param_h: float  # T / n
    param_t: float  # wall-clock: Param_T * n seconds
    k_lo: int  # half-open [k_lo, k_hi)
    k_hi: int
    restart: bool = True
    restart_reconly: bool = True

    def as_metadata(self, n: int) -> dict:
        d = asdict(self)
        d["problem_size"] = n
        d["paper_T"] = self.param_h * n
        return d


# NeurIPS appendix Table tb:sp + README Param_T
_SEARCH_PARAMS: dict[int, SearchParams] = {
    25: SearchParams(
        alpha=0.0,
        beta=10.0,
        max_candidate_num=8,
        param_h=3.0,
        param_t=0.01,
        k_lo=10,
        k_hi=11,
    ),
    50: SearchParams(
        alpha=0.0,
        beta=10.0,
        max_candidate_num=8,
        param_h=3.0,
        param_t=0.01,
        k_lo=5,
        k_hi=15,
    ),
    100: SearchParams(
        alpha=0.0,
        beta=10.0,
        max_candidate_num=8,
        param_h=3.0,
        param_t=0.01,
        k_lo=5,
        k_hi=35,
    ),
    200: SearchParams(
        alpha=0.0,
        beta=10.0,
        max_candidate_num=8,
        param_h=3.0,
        param_t=0.08,
        k_lo=10,
        k_hi=90,
    ),
    500: SearchParams(
        alpha=0.0,
        beta=50.0,
        max_candidate_num=5,
        param_h=2.0,
        param_t=0.04,
        k_lo=30,
        k_hi=130,
    ),
    1000: SearchParams(
        alpha=0.0,
        beta=50.0,
        max_candidate_num=5,
        param_h=2.0,
        param_t=0.04,
        k_lo=10,
        k_hi=110,
    ),
}


def get_search_params(n: int) -> SearchParams:
    """Return paper search params for problem size n.

    Raises:
        KeyError: if n has no configured paper row.
    """
    try:
        return _SEARCH_PARAMS[n]
    except KeyError as e:
        available = sorted(_SEARCH_PARAMS)
        raise KeyError(
            f"No paper search params for size {n}. Available: {available}"
        ) from e
