"""Stage 2 evaluation metrics — within (country_id, year) groups.

Used by LambdaRank training scripts and benchmarks. Global AUC is intentionally
not the primary metric for conditional geographic selection.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _groups_from_frame(df: pd.DataFrame, group_cols: Sequence[str]) -> list[pd.DataFrame]:
    return [g for _, g in df.groupby(list(group_cols), sort=False)]


def ndcg_at_k_within_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
    k_pct: float,
) -> float:
    """Macro-averaged NDCG@k_pct within each group (binary or graded relevance).

    Supports graded relevance labels (e.g., 1–4 from designation event size).
    DCG uses linear gain (rel / log2(rank+2)); IDCG is computed from the ideal
    ranking of the positive items sorted by descending relevance.
    """
    scores: list[float] = []
    offset = 0
    for size in group_sizes:
        if size <= 0:
            continue
        end = offset + int(size)
        yt = y_true[offset:end]
        ys = y_score[offset:end]
        offset = end
        pos_rels = yt[yt > 0]
        if len(pos_rels) == 0:
            continue
        k = max(1, int(np.ceil(size * k_pct / 100.0)))
        # DCG: top-k by predicted score
        order = np.argsort(-ys)
        yt_sorted = yt[order[:k]]
        dcg = sum(float(r) / np.log2(i + 2) for i, r in enumerate(yt_sorted) if r > 0)
        # IDCG: ideal ordering — highest relevance first
        ideal_rels = np.sort(pos_rels)[::-1][:k]
        ideal = sum(float(r) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        if ideal > 0:
            scores.append(dcg / ideal)
    return float(np.mean(scores)) if scores else 0.0


def concordance_within_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
) -> float:
    """Macro-averaged pairwise concordance among (positive, negative) pairs per group."""
    concordances: list[float] = []
    offset = 0
    for size in group_sizes:
        if size <= 0:
            continue
        end = offset + int(size)
        yt = y_true[offset:end]
        ys = y_score[offset:end]
        offset = end
        pos_idx = np.where(yt > 0)[0]
        neg_idx = np.where(yt == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pairs = 0
        concordant = 0
        for pi in pos_idx:
            for ni in neg_idx:
                pairs += 1
                if ys[pi] > ys[ni]:
                    concordant += 1
                elif ys[pi] == ys[ni]:
                    concordant += 0.5
        if pairs > 0:
            concordances.append(concordant / pairs)
    return float(np.mean(concordances)) if concordances else 0.0


def precision_at_k_within_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
    k_pct: float,
) -> float:
    """Macro-averaged precision@k_pct within groups."""
    precs: list[float] = []
    offset = 0
    for size in group_sizes:
        if size <= 0:
            continue
        end = offset + int(size)
        yt = y_true[offset:end]
        ys = y_score[offset:end]
        offset = end
        if (yt > 0).sum() == 0:
            continue
        k = max(1, int(np.ceil(size * k_pct / 100.0)))
        top = np.argsort(-ys)[:k]
        precs.append(float((yt[top] > 0).sum()) / k)
    return float(np.mean(precs)) if precs else 0.0


def lift_at_k_within_groups(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
    k_pct: float,
) -> float:
    """Macro-averaged lift@k_pct within groups."""
    baseline = float(y_true.mean()) if len(y_true) else 0.0
    prec = precision_at_k_within_groups(y_true, y_score, group_sizes, k_pct)
    if baseline <= 0:
        return 0.0
    return prec / baseline


def compute_stage2_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_sizes: np.ndarray,
) -> dict[str, float]:
    """Full Stage 2 metric bundle for a scored, group-sorted array."""
    return {
        "ndcg_at_1pct": ndcg_at_k_within_groups(y_true, y_score, group_sizes, 1.0),
        "ndcg_at_5pct": ndcg_at_k_within_groups(y_true, y_score, group_sizes, 5.0),
        "ndcg_at_10pct": ndcg_at_k_within_groups(y_true, y_score, group_sizes, 10.0),
        "concordance_index_within_groups": concordance_within_groups(
            y_true, y_score, group_sizes
        ),
        "precision_at_1pct_within_groups": precision_at_k_within_groups(
            y_true, y_score, group_sizes, 1.0
        ),
        "precision_at_5pct_within_groups": precision_at_k_within_groups(
            y_true, y_score, group_sizes, 5.0
        ),
        "precision_at_10pct_within_groups": precision_at_k_within_groups(
            y_true, y_score, group_sizes, 10.0
        ),
        "lift_at_1pct_within_groups": lift_at_k_within_groups(
            y_true, y_score, group_sizes, 1.0
        ),
        "lift_at_5pct_within_groups": lift_at_k_within_groups(
            y_true, y_score, group_sizes, 5.0
        ),
        "lift_at_10pct_within_groups": lift_at_k_within_groups(
            y_true, y_score, group_sizes, 10.0
        ),
        "baseline_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "n_groups": int(len(group_sizes)),
        "n_samples": int(len(y_true)),
    }


def compute_stage2_metrics_from_frame(
    df: pd.DataFrame,
    *,
    score_col: str = "y_pred_score",
    label_col: str = "y_true",
    group_cols: Iterable[str] = ("country_id", "year"),
) -> dict[str, float]:
    """Compute metrics from a DataFrame sorted by group_cols."""
    gcols = list(group_cols)
    df = df.sort_values(gcols).reset_index(drop=True)
    group_sizes = df.groupby(gcols, sort=False).size().to_numpy()
    return compute_stage2_metrics(
        df[label_col].to_numpy(dtype=np.float64),
        df[score_col].to_numpy(dtype=np.float64),
        group_sizes,
    )
