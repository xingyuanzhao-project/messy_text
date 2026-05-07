"""
Generate support analysis plots for the annotations-without-supporting-documents
analysis.

Produces the following plots per run configuration:

1. Cumulative support curve: CDF showing the fraction of (victim, model) pairs
   that have reached each support threshold by each turn, broken down by model.
2. Machine classification coverage of human annotations: grouped bar chart
   showing how many victims are fully vs partially machine-supported, grouped
   by how many human labels they have.
3. Joint CDF and annotation coverage: panels 1 and 2 side by side in one row.

Two run configurations are defined in CONFIGS:
- less_labels_multi_models: three selected labels (desenlace, vic_grupo_social,
  captura_tipo) across three models.
- all_labels_one_model: all fifteen codebook labels for a single model.

Commented-out code below main() preserves the turns-to-full-support histogram,
which is redundant with the CDF panel and not included in the draft.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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

THRESHOLD_LAYER_COLORS: List[str] = [
    "#91cc75",
    "#ee6666",
    "#fac858",
    "#73c0de",
    "#9a60b4",
    "#3ba272",
]
THRESHOLD_LINESTYLES: list = [
    "-",           # solid          — Full (100%)
    (0, (8, 2)),   # long dashes    — ≥75%
    (0, (5, 3)),   # medium dashes  — ≥50%
    (0, (3, 3)),   # short dashes   — ≥25%
    ":",           # dotted         — ≥10%
    (0, (1, 1)),   # dense dots     — fallback
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
FIGSIZE_JOINT = (12, 10)
STATS_BOX: Dict[str, Any] = dict(boxstyle="round", facecolor="white", alpha=0.8)
DPI = 150


@dataclass
class RunConfig:
    """All per-run settings for one invocation of the support plots."""

    name: str
    label_keys: List[str]
    support_thresholds: List[float]
    input_path: Path
    output_dir: Path

    @property
    def supported_columns(self) -> Dict[str, str]:
        return {label: f"{label}_supported" for label in self.label_keys}

    @property
    def human_columns(self) -> Dict[str, str]:
        return {label: label for label in self.label_keys}

    @property
    def machine_columns(self) -> Dict[str, str]:
        return {label: f"{label}_classification" for label in self.label_keys}

    @property
    def full_support_threshold(self) -> int:
        return len(self.label_keys)


CONFIGS: List[RunConfig] = [
    RunConfig(
        name="less_labels_multi_models",
        label_keys=["desenlace", "vic_grupo_social", "captura_tipo"],
        support_thresholds=[0.6],
        input_path=(
            ROOT
            / "results_down_sized"
            / "df_text_by_report_classification_consolidated_eval_supported.csv"
        ),
        output_dir=(
            ROOT / "plots" / "evaluation" / "matching" / "less_labels_multi_models"
        ),
    ),
    RunConfig(
        name="all_labels_one_model",
        label_keys=[
            "vic_grupo_social",
            "amenaza_quien",
            "captura_metodo",
            "captura_tipo",
            "cautiverio_trato",
            "desenlace",
            "desenlace_tipo",
            "perp_tipo1",
            "perp_tipo2",
            "proced_contacto1",
            "proced_contacto2",
            "proced_contactado",
            "Tribunal_tipo",
            "proced_sent_tipo",
            "soc_civil",
        ],
        support_thresholds=[0.1, 0.25, 0.5, 0.75],
        input_path=(
            ROOT
            / "df_text_by_report_conversation_classification (2)_eval_supported.csv"
        ),
        output_dir=(
            ROOT / "plots" / "evaluation" / "matching" / "all_labels_one_model"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Section 2: Data loading and helpers
# ---------------------------------------------------------------------------


def load_data(path: Path, cfg: RunConfig) -> pd.DataFrame:
    """
    Read the supported-annotations CSV and validate that all required
    columns are present for the given run configuration.

    Args:
        path: Absolute or relative path to the CSV file.
        cfg: RunConfig specifying which label columns to validate.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
        ValueError: If the loaded CSV is missing one or more required columns.
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
        *cfg.supported_columns.values(),
        *cfg.human_columns.values(),
        *cfg.machine_columns.values(),
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
    """
    if full_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[full_name]
    return full_name.rsplit("/", 1)[-1]


def human_label_count_series(
    df_victims: pd.DataFrame,
    cfg: RunConfig,
) -> pd.Series:
    """
    For each row in a victim-level DataFrame, count how many of the
    human-annotation columns defined in cfg are non-null.

    Args:
        df_victims: DataFrame where each row represents one victim.
        cfg: RunConfig specifying which human label columns to count.

    Returns:
        Integer Series aligned with df_victims.index.
    """
    return df_victims[list(cfg.human_columns.values())].notna().sum(axis=1)


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
            turn_index meeting the threshold.

    Returns:
        DataFrame with columns [victim, model, max_turn, <column_name>].
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


def resolve_thresholds(cfg: RunConfig) -> List[Dict[str, Any]]:
    """
    Build an ordered list of threshold specifications from SUPPORT_THRESHOLDS,
    always including full support (1.0).

    Args:
        cfg: RunConfig used to compute absolute threshold counts from fractions.

    Returns:
        List sorted descending by fraction (strictest first).  Each dict has
        keys: fraction, count, pct, label, label_long, col, color, linestyle.
    """
    fractions = sorted(set(cfg.support_thresholds) | {1.0}, reverse=True)
    specs: List[Dict[str, Any]] = []
    for i, frac in enumerate(fractions):
        count = math.ceil(frac * cfg.full_support_threshold)
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
# Section 3: Draw helpers (draw onto a provided Axes object)
# ---------------------------------------------------------------------------


def _draw_cumulative_support_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    cfg: RunConfig,
) -> None:
    """
    Draw the CDF-style cumulative support curve onto the provided Axes.

    One set of lines per threshold (linestyle), each containing one line per
    model (color) plus an all-models aggregate when multiple models are present.

    Args:
        ax: Matplotlib Axes to draw on.
        df: The full turn-level dataset.
        cfg: RunConfig for threshold computation.
    """
    specs = resolve_thresholds(cfg)
    max_turn = int(df["turn_index"].max())
    turns = np.arange(0, max_turn + 1)
    models = sorted(df["model"].unique())

    pairs_by_spec: Dict[int, pd.DataFrame] = {}
    for spec in specs:
        pairs_by_spec[spec["pct"]] = compute_first_turn_at_threshold(
            df, spec["count"], spec["col"],
        )

    n_total = len(pairs_by_spec[specs[0]["pct"]])

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


def _draw_support_ceiling_by_annotation(
    ax: plt.Axes,
    df: pd.DataFrame,
    cfg: RunConfig,
) -> None:
    """
    Draw the machine classification coverage grouped bar chart onto the
    provided Axes.

    Victims are grouped by how many human-annotated labels they have.
    Within each group, green bars show victims whose machine max_support
    reached n_human_labels; red bars show those it did not.

    Args:
        ax: Matplotlib Axes to draw on.
        df: The full turn-level dataset.
        cfg: RunConfig for human label column selection.
    """
    per_victim = df.drop_duplicates("victim").copy()
    per_victim["n_human_labels"] = human_label_count_series(per_victim, cfg)

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

    bars_s = ax.bar(
        x - bar_width / 2,
        supported_counts,
        bar_width,
        label="Annotations are fully covered",
        color=COLORS["supported"],
        edgecolor="white",
        alpha=0.85,
    )
    bars_n = ax.bar(
        x + bar_width / 2,
        not_supported_counts,
        bar_width,
        label="Annotations are partially covered",
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
    ax.set_xlabel("Annotation availability for each victim")
    ax.set_ylabel("Number of victims")
    ax.set_title("Machine Classification Coverage of Human Annotations")
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


# ---------------------------------------------------------------------------
# Section 4: Public plot functions (each creates its own figure)
# ---------------------------------------------------------------------------


def plot_cumulative_support_curve(
    df: pd.DataFrame,
    output_path: Path,
    cfg: RunConfig,
) -> None:
    """
    CDF-style line plot showing the fraction of (victim, model) pairs
    that have reached each support threshold by each turn_index.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
        cfg: RunConfig for threshold computation.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_CDF)
    _draw_cumulative_support_curve(ax, df, cfg)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_support_ceiling_by_annotation(
    df: pd.DataFrame,
    output_path: Path,
    cfg: RunConfig,
) -> None:
    """
    Grouped bar chart showing victims grouped by how many human labels
    they have, with bars for fully vs partially machine-supported.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
        cfg: RunConfig for human label column selection.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_CEILING)
    _draw_support_ceiling_by_annotation(ax, df, cfg)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_joint_cdf_and_annotation_coverage(
    df: pd.DataFrame,
    output_path: Path,
    cfg: RunConfig,
) -> None:
    """
    Joint figure with two subplots in one row: the cumulative support curve
    on the left and the machine classification coverage of human annotations
    on the right.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
        cfg: RunConfig for both subplots.
    """
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=FIGSIZE_JOINT)
    _draw_cumulative_support_curve(ax_top, df, cfg)
    _draw_support_ceiling_by_annotation(ax_bottom, df, cfg)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_turns_to_full_support(
    df: pd.DataFrame,
    output_path: Path,
    cfg: RunConfig,
) -> None:
    """
    Stacked bar chart of how many turns it takes each (victim, model) pair
    to first reach each support threshold.

    Retained for reference; not called from main() because it is redundant
    with the cumulative support curve panel.

    Args:
        df: The full turn-level dataset.
        output_path: Filesystem path where the PNG will be saved.
        cfg: RunConfig for threshold computation.
    """
    specs = resolve_thresholds(cfg)

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


# ---------------------------------------------------------------------------
# Section 5: Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Load data and generate all support analysis plots for each run config.

    For each config in CONFIGS, produces:
      - cumulative_support_curve.png
      - support_ceiling_by_annotation_availability.png
      - joint_cdf_and_annotation_coverage.png
    """
    for cfg in CONFIGS:
        print(f"\n--- Running: {cfg.name} ---")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        df = load_data(cfg.input_path, cfg)

        # Commented out: turns_to_full_support is redundant with the CDF panel.
        # plot_turns_to_full_support(
        #     df,
        #     cfg.output_dir / "turns_to_full_support.png",
        #     cfg,
        # )
        # print(f"  Saved: {cfg.output_dir / 'turns_to_full_support.png'}")

        plot_cumulative_support_curve(
            df,
            cfg.output_dir / "cumulative_support_curve.png",
            cfg,
        )
        print(f"  Saved: {cfg.output_dir / 'cumulative_support_curve.png'}")

        plot_support_ceiling_by_annotation(
            df,
            cfg.output_dir / "support_ceiling_by_annotation_availability.png",
            cfg,
        )
        print(
            f"  Saved: "
            f"{cfg.output_dir / 'support_ceiling_by_annotation_availability.png'}"
        )

        plot_joint_cdf_and_annotation_coverage(
            df,
            cfg.output_dir / "joint_cdf_and_annotation_coverage.png",
            cfg,
        )
        print(
            f"  Saved: {cfg.output_dir / 'joint_cdf_and_annotation_coverage.png'}"
        )

        print(f"  All plots saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
