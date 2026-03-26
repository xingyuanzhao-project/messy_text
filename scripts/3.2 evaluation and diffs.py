"""
Benchmark new conversational classification results against the old
victim-level baseline and compute per-model improvement diffs.

This script:
1. Loads the new postprocessed classification CSV and the old victim-level
   baseline CSV (``df_text_multi_eval_by_victim.csv``).
2. Takes the old data as-is (no label collapsing). The old baseline is
   the historical reference point and must not be modified.
3. Collapses the new results to the final turn per ``(victim, model)``.
4. Computes per-model, per-label accuracy on the **full** old dataset
   and the **full** new final-turn dataset independently.
5. Computes the diff as ``new_accuracy - old_accuracy`` per model per
   label.
6. Writes benchmark CSVs, diff CSVs, per-label plots, per-model plots,
   and a human-readable summary.

Only accuracy is reported. The old and new systems operate on different
label spaces (old: fine-grained original categories; new: collapsed
postprocessed categories), so the comparison shows total end-to-end
improvement including the label-space simplification.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

NEW_INPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated_eval_supported.csv"
)
OLD_INPUT_PATH = ROOT / "df_text_multi_eval_by_victim.csv"
TAXONOMY_PATH = ROOT / "config" / "taxonomy.json"

OUTPUT_DIR = ROOT / "results_down_sized" / "evaluation_diffs"
NEW_BENCHMARKS_PATH = OUTPUT_DIR / "benchmarks_new_by_model_and_label.csv"
OLD_BENCHMARKS_PATH = OUTPUT_DIR / "benchmarks_old_by_model_and_label.csv"
DIFFS_PATH = OUTPUT_DIR / "diffs_by_model_and_label.csv"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.txt"

PLOT_DIR = ROOT / "plots" / "post_processing_v3"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

VICTIM_COL = "victim"
MODEL_COL = "model"
INDEX_COL = "index"
TURN_INDEX_COL = "turn_index"

LABEL_ORDER = ["vic_grupo_social", "captura_tipo", "desenlace"]
CLASSIFICATION_SUFFIX = "_classification"

# ---------------------------------------------------------------------------
# Plot style constants (matching plots/post_processing_v2)
# ---------------------------------------------------------------------------

COLOR_OLD = "#4ecdc4"
COLOR_NEW = "#45b7d1"
COLOR_DIFF_POS = "green"
COLOR_DIFF_NEG = "red"
BAR_WIDTH = 0.25
BAR_ALPHA = 0.8
ANNOTATION_FONTSIZE = 9
PLOT_DPI = 150

# ---------------------------------------------------------------------------
# Short model names for filenames
# ---------------------------------------------------------------------------

MODEL_SHORT_NAMES = {
    "gaunernst/gemma-3-12b-it-int4-awq": "gemma",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "llama",
    "mistralai/Ministral-3-8B-Instruct-2512": "mistral",
}

MODEL_DISPLAY_NAMES = {
    "gaunernst/gemma-3-12b-it-int4-awq": "Gemma-3-12b",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4": "Llama-3.1-8B",
    "mistralai/Ministral-3-8B-Instruct-2512": "Ministral-3-8B",
}

LABEL_DISPLAY_NAMES = {
    "vic_grupo_social": "Social group",
    "captura_tipo": "Place of disappearance",
    "desenlace": "Outcome",
}

LEGEND_OLD = "Simple Summarization"
LEGEND_NEW = "Extractive Summarization"
LEGEND_DIFF = "Diff"

MODEL_ORDER = [
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "gaunernst/gemma-3-12b-it-int4-awq",
    "mistralai/Ministral-3-8B-Instruct-2512",
]

LABEL_TITLES = {
    "vic_grupo_social": "Social group membership (Accuracy)",
    "captura_tipo": "Type of place of the disappearance (Accuracy)",
    "desenlace": "Outcome of the disappearance (Accuracy)",
}


def display_path(path: Path) -> str:
    """Return a workspace-relative display string when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return the parsed dict."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {display_path(path)}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV as a DataFrame of string objects."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {display_path(path)}")
    return pd.read_csv(path, encoding="utf-8", dtype=object)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV with stable newline convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """Raise if any required columns are missing from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Final-turn selection
# ---------------------------------------------------------------------------


def select_final_turn(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the row with the highest turn_index per (victim, model).

    Returns a copy with one row per ``(victim, model)`` pair.
    """
    df = df.copy()
    df[TURN_INDEX_COL] = pd.to_numeric(df[TURN_INDEX_COL], errors="coerce")
    missing = int(df[TURN_INDEX_COL].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} rows have missing turn_index; cannot select final turn."
        )
    df = df.sort_values(
        [MODEL_COL, VICTIM_COL, TURN_INDEX_COL, INDEX_COL], kind="stable"
    )
    df = (
        df.groupby([MODEL_COL, VICTIM_COL], dropna=False, as_index=False)
        .tail(1)
        .copy()
    )
    dupes = df.duplicated(subset=[MODEL_COL, VICTIM_COL], keep=False)
    if dupes.any():
        raise ValueError("Final-turn selection left duplicate (model, victim) rows.")
    return df


# ---------------------------------------------------------------------------
# Accuracy computation
# ---------------------------------------------------------------------------


def compute_accuracy(
    df: pd.DataFrame,
    label_col: str,
    cls_col: str,
) -> float:
    """Compute accuracy for one label column pair in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *label_col* (human annotation) and *cls_col*
        (machine classification).
    label_col : str
        Column name for the ground-truth human annotation.
    cls_col : str
        Column name for the machine classification.

    Returns
    -------
    float
        Accuracy (fraction of rows where human == machine).
    """
    y_true = df[label_col].fillna("").astype(str).str.strip()
    y_pred = df[cls_col].fillna("").astype(str).str.strip()
    n = len(y_true)
    if n == 0:
        raise ValueError(f"Cannot compute accuracy on 0 rows for {label_col}.")
    return int((y_true == y_pred).sum()) / n


# ---------------------------------------------------------------------------
# Benchmark computation
# ---------------------------------------------------------------------------


def compute_benchmarks(
    df: pd.DataFrame,
    dataset_tag: str,
) -> pd.DataFrame:
    """Compute per-model, per-label accuracy on all rows in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``model``, label columns, and classification columns.
    dataset_tag : str
        Human-readable tag used in printed diagnostics (e.g. ``"old"``).

    Returns
    -------
    pd.DataFrame
        One row per ``(model, label)`` with an ``Accuracy`` column.
    """
    rows: list[dict[str, Any]] = []
    for model in sorted(df[MODEL_COL].dropna().unique()):
        df_model = df[df[MODEL_COL] == model]
        for label in LABEL_ORDER:
            cls_col = f"{label}{CLASSIFICATION_SUFFIX}"
            acc = compute_accuracy(df_model, label, cls_col)
            rows.append({
                "model": model,
                "label": label,
                "rows": len(df_model),
                "Accuracy": acc,
            })
    result = pd.DataFrame(rows)
    print(f"  [{dataset_tag}] computed {len(result)} benchmark rows")
    return result


def compute_diffs(
    old_bench: pd.DataFrame,
    new_bench: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accuracy diffs as ``new - old`` per (model, label).

    Old and new benchmarks are each computed on their full respective
    datasets (different N is expected). The diff is a simple subtraction
    of the aggregate accuracy values.

    Parameters
    ----------
    old_bench, new_bench : pd.DataFrame
        Output of ``compute_benchmarks``.

    Returns
    -------
    pd.DataFrame
        One row per ``(model, label)`` with old accuracy, new accuracy,
        delta, and both row counts.
    """
    merged = old_bench.merge(
        new_bench,
        on=["model", "label"],
        suffixes=("_old", "_new"),
        how="inner",
    )
    merged["Delta Accuracy"] = merged["Accuracy_new"] - merged["Accuracy_old"]
    merged = merged.rename(columns={
        "Accuracy_old": "old_Accuracy",
        "Accuracy_new": "new_Accuracy",
        "rows_old": "old_rows",
        "rows_new": "new_rows",
    })
    return merged[
        ["model", "label", "old_rows", "new_rows",
         "old_Accuracy", "new_Accuracy", "Delta Accuracy"]
    ]


# ---------------------------------------------------------------------------
# Per-label plots (one PNG per label, subplots per model)
# ---------------------------------------------------------------------------


def _annotate_bars(ax: plt.Axes, bars, values: list[float], is_diff: bool) -> None:
    """Add value labels above each bar."""
    for bar, val in zip(bars, values):
        y = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.04
        va = "bottom" if val >= 0 else "top"
        txt = f"{val:+.2f}" if is_diff else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y, txt,
            ha="center", va=va,
            fontsize=ANNOTATION_FONTSIZE, fontweight="bold",
        )


def plot_per_label(
    label: str,
    context_def: str,
    old_bench: pd.DataFrame,
    new_bench: pd.DataFrame,
    diffs: pd.DataFrame,
) -> Path:
    """Create one unified PNG per label with all models on one axis.

    The x-axis has one group per model (3 groups, 3 bars each = 9 bars).

    Parameters
    ----------
    label : str
        Label name (e.g. ``"desenlace"``).
    context_def : str
        Taxonomy context definition shown as footnote.
    old_bench, new_bench : pd.DataFrame
        Full-data benchmark tables (from ``compute_benchmarks``).
    diffs : pd.DataFrame
        Diff table (from ``compute_diffs``).

    Returns
    -------
    Path
        Saved PNG path.
    """
    old_lab = old_bench[old_bench["label"] == label]
    new_lab = new_bench[new_bench["label"] == label]
    diff_lab = diffs[diffs["label"] == label]
    available = set(new_lab["model"].unique())
    models = [m for m in MODEL_ORDER if m in available]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))

    o_vals, n_vals, d_vals = [], [], []
    for model in models:
        o = old_lab[old_lab["model"] == model]
        n = new_lab[new_lab["model"] == model]
        d = diff_lab[diff_lab["model"] == model]
        o_vals.append(float(o.iloc[0]["Accuracy"]) if not o.empty else 0.0)
        n_vals.append(float(n.iloc[0]["Accuracy"]) if not n.empty else 0.0)
        d_vals.append(float(d.iloc[0]["Delta Accuracy"]) if not d.empty else 0.0)

    bars_o = ax.bar(
        x - BAR_WIDTH, o_vals, BAR_WIDTH,
        label=LEGEND_OLD, color=COLOR_OLD,
        edgecolor="black", alpha=BAR_ALPHA,
    )
    bars_n = ax.bar(
        x, n_vals, BAR_WIDTH,
        label=LEGEND_NEW, color=COLOR_NEW,
        edgecolor="black", alpha=BAR_ALPHA,
    )
    d_colors = [COLOR_DIFF_POS if v >= 0 else COLOR_DIFF_NEG for v in d_vals]
    bars_d = ax.bar(
        x + BAR_WIDTH, d_vals, BAR_WIDTH,
        label=LEGEND_DIFF, color=d_colors,
        edgecolor="black", alpha=BAR_ALPHA,
    )

    _annotate_bars(ax, bars_o, o_vals, False)
    _annotate_bars(ax, bars_n, n_vals, False)
    _annotate_bars(ax, bars_d, d_vals, True)

    ax.set_ylim(-0.40, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models], fontsize=10,
    )
    ax.set_ylabel("Accuracy")
    ax.axhline(y=0, color="black", linewidth=0.7)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    title = LABEL_TITLES.get(label, f"{label} (Accuracy)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = PLOT_DIR / f"metrics_{label}.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Per-model plots (one PNG per model, one subplot per label)
# ---------------------------------------------------------------------------


def plot_per_model(
    model: str,
    old_bench: pd.DataFrame,
    new_bench: pd.DataFrame,
    diffs: pd.DataFrame,
) -> Path:
    """Create one unified PNG per model with all labels on one axis.

    The x-axis has one group per label (3 groups, 3 bars each = 9 bars).

    Parameters
    ----------
    model : str
        Full model identifier string.
    old_bench, new_bench : pd.DataFrame
        Full-data benchmark tables.
    diffs : pd.DataFrame
        Diff table.

    Returns
    -------
    Path
        Saved PNG path.
    """
    o_model = old_bench[old_bench["model"] == model]
    n_model = new_bench[new_bench["model"] == model]
    d_model = diffs[diffs["model"] == model]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(LABEL_ORDER))

    o_vals, n_vals, d_vals = [], [], []
    for label in LABEL_ORDER:
        o_row = o_model[o_model["label"] == label]
        n_row = n_model[n_model["label"] == label]
        d_row = d_model[d_model["label"] == label]
        o_vals.append(float(o_row.iloc[0]["Accuracy"]) if not o_row.empty else 0.0)
        n_vals.append(float(n_row.iloc[0]["Accuracy"]) if not n_row.empty else 0.0)
        d_vals.append(float(d_row.iloc[0]["Delta Accuracy"]) if not d_row.empty else 0.0)

    bars_o = ax.bar(
        x - BAR_WIDTH, o_vals, BAR_WIDTH,
        label=LEGEND_OLD, color=COLOR_OLD,
        edgecolor="black", alpha=BAR_ALPHA,
    )
    bars_n = ax.bar(
        x, n_vals, BAR_WIDTH,
        label=LEGEND_NEW, color=COLOR_NEW,
        edgecolor="black", alpha=BAR_ALPHA,
    )
    d_colors = [COLOR_DIFF_POS if v >= 0 else COLOR_DIFF_NEG for v in d_vals]
    bars_d = ax.bar(
        x + BAR_WIDTH, d_vals, BAR_WIDTH,
        label=LEGEND_DIFF, color=d_colors,
        edgecolor="black", alpha=BAR_ALPHA,
    )

    _annotate_bars(ax, bars_o, o_vals, False)
    _annotate_bars(ax, bars_n, n_vals, False)
    _annotate_bars(ax, bars_d, d_vals, True)

    ax.set_ylim(-0.40, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [LABEL_DISPLAY_NAMES.get(l, l) for l in LABEL_ORDER], fontsize=10,
    )
    ax.set_ylabel("Accuracy")
    ax.axhline(y=0, color="black", linewidth=0.7)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    display_model = MODEL_DISPLAY_NAMES.get(model, model)
    ax.set_title(f"Accuracy: {display_model}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    short = MODEL_SHORT_NAMES.get(model, model.split("/")[-1])
    out = PLOT_DIR / f"metrics_{short}.png"
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------


def build_summary_text(
    df_old: pd.DataFrame,
    df_new_raw: pd.DataFrame,
    df_new_final: pd.DataFrame,
    old_bench: pd.DataFrame,
    new_bench: pd.DataFrame,
    diffs: pd.DataFrame,
    plot_paths: list[Path],
) -> str:
    """Build the full human-readable summary string."""
    lines: list[str] = []
    lines.append("=== DATA SUMMARY ===")
    lines.append(f"Old rows loaded: {len(df_old)}")
    lines.append(f"New rows loaded (raw): {len(df_new_raw)}")
    lines.append(f"New rows after final-turn collapse: {len(df_new_final)}")
    lines.append(
        f"Models evaluated: "
        f"{', '.join(sorted(df_new_final[MODEL_COL].dropna().unique()))}"
    )
    lines.append("")

    lines.append("=== OLD BENCHMARKS (full old data per model, as-is) ===")
    for _, row in old_bench.iterrows():
        lines.append(
            f"  {row['model']}, {row['label']} "
            f"(N={int(row['rows'])}): Accuracy={row['Accuracy']:.3f}"
        )
    lines.append("")

    lines.append("=== NEW BENCHMARKS (full new final-turn data per model) ===")
    for _, row in new_bench.iterrows():
        lines.append(
            f"  {row['model']}, {row['label']} "
            f"(N={int(row['rows'])}): Accuracy={row['Accuracy']:.3f}"
        )
    lines.append("")

    lines.append("=== DIFFS (new_accuracy - old_accuracy) ===")
    for _, row in diffs.iterrows():
        lines.append(
            f"  {row['model']}, {row['label']}: "
            f"old={row['old_Accuracy']:.3f} (N={int(row['old_rows'])}), "
            f"new={row['new_Accuracy']:.3f} (N={int(row['new_rows'])}), "
            f"Delta={row['Delta Accuracy']:+.3f}"
        )
    lines.append("")

    lines.append("=== PLOT FILES ===")
    for p in plot_paths:
        lines.append(f"  {display_path(p)}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_evaluation_and_diffs() -> dict[str, Any]:
    """Run the full evaluation pipeline and return all result objects."""

    taxonomy = load_json(TAXONOMY_PATH)
    context_defs = taxonomy.get("context_definitions", {})

    df_old = load_csv(OLD_INPUT_PATH)
    df_new_raw = load_csv(NEW_INPUT_PATH)

    print(f"Loaded old CSV: {len(df_old)} rows from {display_path(OLD_INPUT_PATH)}")
    print(f"Loaded new CSV: {len(df_new_raw)} rows from {display_path(NEW_INPUT_PATH)}")

    label_cols = LABEL_ORDER + [f"{l}{CLASSIFICATION_SUFFIX}" for l in LABEL_ORDER]
    validate_columns(df_old, [VICTIM_COL, MODEL_COL] + label_cols, "old CSV")
    validate_columns(
        df_new_raw,
        [VICTIM_COL, MODEL_COL, TURN_INDEX_COL, INDEX_COL] + label_cols,
        "new CSV",
    )

    print("Selecting final turn per (victim, model) in new data...")
    df_new_final = select_final_turn(df_new_raw)
    print(f"  {len(df_new_raw)} -> {len(df_new_final)} rows after final-turn collapse")

    print("Computing OLD benchmarks (full old data per model, as-is):")
    old_bench = compute_benchmarks(df_old, "old")

    print("Computing NEW benchmarks (full new final-turn data per model):")
    new_bench = compute_benchmarks(df_new_final, "new")

    print("Computing diffs (new - old):")
    diffs = compute_diffs(old_bench, new_bench)
    for _, row in diffs.iterrows():
        print(
            f"  {row['model']}, {row['label']}: "
            f"old={row['old_Accuracy']:.3f}, new={row['new_Accuracy']:.3f}, "
            f"Delta={row['Delta Accuracy']:+.3f}"
        )

    write_csv(old_bench, OLD_BENCHMARKS_PATH)
    write_csv(new_bench, NEW_BENCHMARKS_PATH)
    write_csv(diffs, DIFFS_PATH)
    print(f"Wrote benchmarks to {display_path(OUTPUT_DIR)}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []

    for label in LABEL_ORDER:
        p = plot_per_label(
            label=label,
            context_def=str(context_defs.get(label, "")),
            old_bench=old_bench,
            new_bench=new_bench,
            diffs=diffs,
        )
        plot_paths.append(p)
        print(f"  Saved per-label plot: {display_path(p)}")

    for model in MODEL_ORDER:
        p = plot_per_model(
            model=model,
            old_bench=old_bench,
            new_bench=new_bench,
            diffs=diffs,
        )
        plot_paths.append(p)
        print(f"  Saved per-model plot: {display_path(p)}")

    summary_text = build_summary_text(
        df_old=df_old,
        df_new_raw=df_new_raw,
        df_new_final=df_new_final,
        old_bench=old_bench,
        new_bench=new_bench,
        diffs=diffs,
        plot_paths=plot_paths,
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(summary_text)
    print(f"Wrote summary to {display_path(SUMMARY_PATH)}")

    return {
        "old_bench": old_bench,
        "new_bench": new_bench,
        "diffs": diffs,
        "summary_text": summary_text,
        "plot_paths": plot_paths,
    }


def main() -> int:
    """Entry point: run the pipeline and print summary to console."""
    result = run_evaluation_and_diffs()
    print("")
    print(result["summary_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
