"""
Generate support analysis plots for the annotations-without-supporting-documents
analysis.

Produces three plots that visualize how human-coded annotations align with
machine-produced classifications in terms of document availability:

1. Turns to full support: histogram of how many conversation turns it takes
   for a (victim, model) pair to achieve support for all three labels.
2. Cumulative support curve: CDF showing the fraction of pairs that have
   reached full support by each turn, broken down by model.
3. Support ceiling by human annotation availability: grouped bar chart
   showing how many victims are fully machine-supported vs not, grouped
   by how many human labels they have.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Section 1: Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated_eval.csv.csv"
)
OUTPUT_DIR = ROOT / "plots" / "post_processing_v3" / "supports"

LABEL_KEYS: List[str] = [
    "desenlace",
    "vic_grupo_social",
    "captura_tipo",
]

SUPPORTED_COLUMNS: Dict[str, str] = {
    label: f"{label}_supported" for label in LABEL_KEYS
}

HUMAN_COLUMNS: Dict[str, str] = {label: label for label in LABEL_KEYS}

MACHINE_COLUMNS: Dict[str, str] = {
    label: f"{label}_classification" for label in LABEL_KEYS
}

FULL_SUPPORT_THRESHOLD: int = len(LABEL_KEYS)
SUPPORT_THRESHOLDS: List[float] = [0.6]

THRESHOLD_LAYER_COLORS: List[str] = [
    "#91cc75",
    "#ee6666",
    "#fac858",
    "#73c0de",
]
THRESHOLD_LINESTYLES: list = [
    "-",
    (0, (5, 2)),
    "--",
    (0, (5, 5)),
    "-.",
    (0, (1, 1)),
]

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "mistralai/Ministral-3-8B-Instruct-2512": "Mistral 8B",
    "gaunernst/gemma-3-12b-it-int4-awq": "Gemma 12B",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "Llama 8B",
}

COLORS: Dict[str, str] = {
    "supported": "#91cc75",
    "not_supported": "#ee6666",
    "all_models": "#333333",
    "mistralai/Ministral-3-8B-Instruct-2512": "#5470c6",
    "gaunernst/gemma-3-12b-it-int4-awq": "#91cc75",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "#fac858",
}

FIGSIZE_HISTOGRAM = (10, 5)
FIGSIZE_CDF = (10, 5)
FIGSIZE_CEILING = (10, 5)
STATS_BOX: Dict[str, Any] = dict(boxstyle="round", facecolor="white", alpha=0.8)
DPI = 150


# ---------------------------------------------------------------------------
# Section 2: Data loading and helpers
# ---------------------------------------------------------------------------


def load_data(path: Path) -> pd.DataFrame:
    """
    Read the supported-annotations CSV and validate that all required
    columns are present.

    Args:
        path: Absolute or relative path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
        ValueError: If the loaded CSV is missing one or more required
            columns (victim, model, turn_index, support_count, the per-label
            supported flags, and the human/machine label columns).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")

    required = {
        "victim",
        "model",
        "turn_index",
        "support_count",
        "any_label_supported",
        *SUPPORTED_COLUMNS.values(),
        *HUMAN_COLUMNS.values(),
        *MACHINE_COLUMNS.values(),
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}"
        )

    return df


def model_display_name(full_name: str) -> str:
    """
    Return a short display name for a model, falling back to the last
    path component if no explicit mapping exists in MODEL_DISPLAY_NAMES.

    Args:
        full_name: Full HuggingFace-style model identifier
            (e.g. ``"mistralai/Ministral-3-8B-Instruct-2512"``).

    Returns:
        Short display string suitable for axis labels and legends.
    """
    if full_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[full_name]
    return full_name.rsplit("/", 1)[-1]


def human_label_count_series(df_victims: pd.DataFrame) -> pd.Series:
    """
    For each row in a victim-level DataFrame, count how many of the three
    human-annotation columns (desenlace, vic_grupo_social, captura_tipo)
    are non-null.

    Args:
        df_victims: DataFrame where each row represents one victim.  Must
            contain the columns listed in HUMAN_COLUMNS.

    Returns:
        Integer Series aligned with df_victims.index, values in [0, 3].
    """
    return df_victims[list(HUMAN_COLUMNS.values())].notna().sum(axis=1)


def compute_first_turn_at_threshold(
    df: pd.DataFrame,
    threshold: int,
    column_name: str,
) -> pd.DataFrame:
    """
    For each (victim, model) pair, find the earliest turn_index at which
    support_count reaches a given threshold.

    Args:
        df: The full turn-level dataset.  Must contain columns
            ``victim``, ``model``, ``turn_index``, and ``support_count``.
        threshold: The minimum support_count value that qualifies as
            having reached the threshold.
        column_name: Name for the output column that holds the first
            turn_index meeting the threshold (e.g. ``"first_full_turn"``).

    Returns:
        DataFrame with columns [victim, model, max_turn, <column_name>].
            max_turn is the last turn_index available for that pair.
            <column_name> is the earliest turn at which support_count
            reached the threshold, or NaN if it never did.
    """
    pairs = (
        df.groupby(["victim", "model"])["turn_index"]
        .max()
        .reset_index()
        .rename(columns={"turn_index": "max_turn"})
    )

    qualifying = df[df["support_count"] >= threshold]
    first_turn = (
        qualifying
        .groupby(["victim", "model"])["turn_index"]
        .min()
        .reset_index()
        .rename(columns={"turn_index": column_name})
    )

    return pairs.merge(first_turn, on=["victim", "model"], how="left")


def resolve_thresholds() -> List[Dict[str, Any]]:
    """
    Build an ordered list of threshold specifications from
    SUPPORT_THRESHOLDS, always including full support (1.0).

    Returns:
        List sorted descending by fraction (strictest first).  Each dict
        has keys: fraction, count, pct, label, col, color, linestyle.
    """
    fractions = sorted(set(SUPPORT_THRESHOLDS) | {1.0}, reverse=True)
    specs: List[Dict[str, Any]] = []
    for i, frac in enumerate(fractions):
        count = math.ceil(frac * FULL_SUPPORT_THRESHOLD)
        pct = int(round(frac * 100))
        if frac >= 1.0:
            label = "Full"
            label_long = "Full support"
        else:
            label = f">= {pct}%"
            label_long = f">= {pct}% support"
        specs.append({
            "fraction": frac,
            "count": count,
            "pct": pct,
            "label": label,
            "label_long": label_long,
            "col": f"first_turn_{pct}pct",
            "color": THRESHOLD_LAYER_COLORS[i % len(THRESHOLD_LAYER_COLORS)],
            "linestyle": THRESHOLD_LINESTYLES[i % len(THRESHOLD_LINESTYLES)],
        })
    return specs


# ---------------------------------------------------------------------------
# Section 3: Plot functions
# ---------------------------------------------------------------------------


def plot_turns_to_full_support(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Stacked bar chart of how many turns it takes each (victim, model) pair
    to first reach each support threshold defined in SUPPORT_THRESHOLDS
    (plus full support at 1.0).

    Layers are stacked from strictest (bottom) to least strict (top).
    At a given turn position a pair is attributed to the strictest
    threshold it first reaches at that turn, avoiding double-counting.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
    """
    specs = resolve_thresholds()

    pairs_base = (
        df.groupby(["victim", "model"])["turn_index"]
        .max()
        .reset_index()
        .rename(columns={"turn_index": "max_turn"})
    )
    for spec in specs:
        ft = compute_first_turn_at_threshold(df, spec["count"], spec["col"])
        pairs_base = pairs_base.merge(
            ft[["victim", "model", spec["col"]]],
            on=["victim", "model"],
            how="left",
        )

    all_first = pd.concat(
        [pairs_base[s["col"]].dropna() for s in specs]
    )
    max_turn = int(all_first.max()) if len(all_first) > 0 else 0
    turn_range = np.arange(0, max_turn + 1)

    for i, spec in enumerate(specs):
        layer = pairs_base[spec["col"]].copy()
        for j in range(i):
            same_turn = (
                (pairs_base[specs[j]["col"]] == pairs_base[spec["col"]])
                & pairs_base[specs[j]["col"]].notna()
            )
            layer = layer.where(~same_turn, other=np.nan)
        spec["layer_counts"] = np.array([
            int((layer == t).sum()) for t in turn_range
        ])

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    bottom = np.zeros(len(turn_range))
    for spec in specs:
        ax.bar(
            turn_range,
            spec["layer_counts"],
            bottom=bottom,
            color=spec["color"],
            edgecolor="white",
            alpha=0.85,
            label=spec["label"],
        )
        bottom = bottom + spec["layer_counts"]

    for t_idx in range(len(turn_range)):
        total_height = int(bottom[t_idx])
        if total_height > 0:
            ax.text(
                t_idx, total_height, f"{total_height}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(turn_range)
    ax.set_xticklabels([str(t) for t in turn_range])
    ax.set_xlabel("Turn index when % of supports reached")
    ax.set_ylabel("Count of the victim-model pairs")
    ax.set_title("Turns to support by threshold")
    ax.legend(loc="upper right")

    total = len(pairs_base)
    stats_lines = []
    for spec in specs:
        n = int(pairs_base[spec["col"]].notna().sum())
        stats_lines.append(
            f"Eventually reached {spec['label_long'].lower()}: "
            f"{n:,} ({n / total:.0%})"
        )
    ax.text(
        0.97, 0.70, "\n".join(stats_lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=STATS_BOX,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_support_curve(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    CDF-style line plot showing the fraction of (victim, model) pairs
    that have reached each support threshold by each turn_index.

    One set of lines per threshold (distinguished by linestyle), each
    containing one line per model (distinguished by color) plus an
    all-models aggregate.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
    """
    specs = resolve_thresholds()
    max_turn = int(df["turn_index"].max())
    turns = np.arange(0, max_turn + 1)
    models = sorted(df["model"].unique())

    pairs_by_spec: Dict[int, pd.DataFrame] = {}
    for spec in specs:
        pairs_by_spec[spec["pct"]] = compute_first_turn_at_threshold(
            df, spec["count"], spec["col"],
        )

    n_total = len(pairs_by_spec[specs[0]["pct"]])
    fig, ax = plt.subplots(figsize=FIGSIZE_CDF)

    for spec in specs:
        pairs = pairs_by_spec[spec["pct"]]
        col = spec["col"]
        is_full = spec["fraction"] >= 1.0

        single_model = len(models) == 1
        for model_name in models:
            mp = pairs[pairs["model"] == model_name]
            n_m = len(mp)
            frac = np.array([
                (mp[col] <= t).sum() / n_m if n_m > 0 else 0.0
                for t in turns
            ])
            display = model_display_name(model_name)
            line_color = "#000000" if single_model else COLORS.get(model_name)
            ax.plot(
                turns, frac,
                color=line_color,
                linewidth=1.5,
                alpha=0.8 if is_full else 0.5,
                linestyle=spec["linestyle"],
                label=f"{display}, {spec['label_long']}",
            )

        if len(models) > 1:
            frac_all = np.array([
                (pairs[col] <= t).sum() / n_total if n_total > 0 else 0.0
                for t in turns
            ])
            ax.plot(
                turns, frac_all,
                color=COLORS["all_models"],
                linewidth=2.5,
                linestyle=spec["linestyle"],
                alpha=0.9 if is_full else 0.6,
                label=f"Average, {spec['label_long']}",
            )

    ax.set_xlabel("Turn index")
    ax.set_ylabel("% of (victim, model) pairs reaching threshold")
    ax.set_title("Cumulative support by turn")
    ax.set_xlim(-0.2, max_turn + 0.2)
    ax.set_ylim(0, None)
    ax.legend(loc="center right", fontsize=8, ncol=1)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_support_ceiling_by_annotation(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Grouped bar chart showing victims grouped by how many human labels
    they have (1, 2, or 3), with bars for "fully machine-supported" vs
    "not fully machine-supported" within each group.

    A victim is considered fully machine-supported when, across all their
    rows (any model, any turn), the maximum support_count they achieve is
    at least equal to the number of human-annotated labels they have.

    Victims with zero human labels are excluded from the plot (they have
    nothing to support) but their count is reported in the stats box.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
    """
    per_victim = df.drop_duplicates("victim").copy()
    per_victim["n_human_labels"] = human_label_count_series(per_victim)

    max_support = (
        df.groupby("victim")["support_count"]
        .max()
        .reset_index()
        .rename(columns={"support_count": "max_support"})
    )
    per_victim = per_victim.merge(max_support, on="victim", how="left")
    per_victim["fully_supported"] = (
        per_victim["max_support"] >= per_victim["n_human_labels"]
    )

    n_zero_labels = int((per_victim["n_human_labels"] == 0).sum())
    annotated = per_victim[per_victim["n_human_labels"] > 0]
    groups = sorted(annotated["n_human_labels"].unique())

    supported_counts: List[int] = []
    not_supported_counts: List[int] = []
    for g in groups:
        subset = annotated[annotated["n_human_labels"] == g]
        n_sup = int(subset["fully_supported"].sum())
        supported_counts.append(n_sup)
        not_supported_counts.append(len(subset) - n_sup)

    x = np.arange(len(groups))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_CEILING)

    bars_s = ax.bar(
        x - bar_width / 2,
        supported_counts,
        bar_width,
        label="Full coverage of documents",
        color=COLORS["supported"],
        edgecolor="white",
        alpha=0.85,
    )
    bars_n = ax.bar(
        x + bar_width / 2,
        not_supported_counts,
        bar_width,
        label="Partial coverage of documents",
        color=COLORS["not_supported"],
        edgecolor="white",
        alpha=0.85,
    )

    for bar, count in zip(bars_s, supported_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, count in zip(bars_n, not_supported_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{g} label{'s' if g > 1 else ''}" for g in groups])
    ax.set_xlabel("Number of human-annotated labels per victim")
    ax.set_ylabel("Number of victims")
    ax.set_title("Support ceiling by human annotation availability")
    ax.legend(loc="upper left")

    total_annotated = len(annotated)
    total_supported = int(annotated["fully_supported"].sum())
    stats_text = (
        f"Victims with annotations: {total_annotated:,}\n"
        f"Full coverage: {total_supported:,} "
        f"({total_supported / total_annotated:.0%})\n"
        f"Partial coverage: {total_annotated - total_supported:,} "
        f"({(total_annotated - total_supported) / total_annotated:.0%})\n"
        f"Excluded (no labels): {n_zero_labels:,}"
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=STATS_BOX,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 4: Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load data, generate all support analysis plots, and print output paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT_PATH)

    plot_turns_to_full_support(
        df,
        OUTPUT_DIR / "turns_to_full_support.png",
    )
    print(f"  Saved: {OUTPUT_DIR / 'turns_to_full_support.png'}")

    plot_cumulative_support_curve(
        df,
        OUTPUT_DIR / "cumulative_support_curve.png",
    )
    print(f"  Saved: {OUTPUT_DIR / 'cumulative_support_curve.png'}")

    plot_support_ceiling_by_annotation(
        df,
        OUTPUT_DIR / "support_ceiling_by_annotation_availability.png",
    )
    print(f"  Saved: {OUTPUT_DIR / 'support_ceiling_by_annotation_availability.png'}")

    print(f"\nAll support analysis plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
