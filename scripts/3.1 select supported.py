"""
Select supported rows from the postprocessed classification CSV.

A row is "supported" for a given label when both the human annotation
column (e.g. ``desenlace``) and the machine classification column
(e.g. ``desenlace_classification``) are non-NaN on the same row.

Behaviour is controlled by three feature toggles:

``KEEP_ONLY_SUPPORTED_ROWS``
    When ``True``, drop every row where no label pair is supported
    (i.e. the row contributes nothing to any per-label evaluation).
    When ``False``, all rows are kept regardless of support status.

``FLAG_SUPPORTS``
    When ``True``, append per-label boolean ``{base}_supported``
    columns, an ``any_label_supported`` boolean, and a
    ``support_count`` integer column that counts how many label pairs
    are supported on each row.  When ``False`` these columns are
    omitted from the output.

``MASK_UNSUPPORTED``
    When ``True``, for every unsupported label pair (human annotation
    is non-NaN but machine classification is NaN) the human annotation
    value is replaced with an empty string.  This prevents downstream
    accuracy calculations from penalising the machine for documents it
    never saw.  When ``False`` no values are altered.

The label bases are read from ``config/taxonomy.json`` so adding or
removing labels requires zero code changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated_eval.csv"
)
OUTPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated_eval_supported.csv"
)
TAXONOMY_PATH = ROOT / "config" / "taxonomy.json"

# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------

KEEP_ONLY_SUPPORTED_ROWS = False
FLAG_SUPPORTS = True
MASK_UNSUPPORTED = False

# ---------------------------------------------------------------------------
# Column naming conventions
# ---------------------------------------------------------------------------

CLASSIFICATION_SUFFIX = "_classification"
SUPPORTED_SUFFIX = "_supported"
ANY_SUPPORTED_COLUMN = "any_label_supported"
SUPPORT_COUNT_COLUMN = "support_count"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def display_path(path: Path) -> str:
    """Return a workspace-relative display string, falling back to absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_input_frame(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, raising if the file does not exist.

    Parameters
    ----------
    path : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded frame with all columns read as object dtype.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {display_path(path)}"
        )
    return pd.read_csv(path, encoding="utf-8", dtype=object)


def get_category_bases(taxonomy_path: Path) -> list[str]:
    """Extract the category base names from the taxonomy JSON.

    Reads the ``label_options`` mapping from *taxonomy_path* and returns
    its keys as an ordered list.  Raises if the file is missing or the
    mapping is empty.

    Parameters
    ----------
    taxonomy_path : Path
        Absolute path to the taxonomy JSON file
        (e.g. ``config/taxonomy.json``).

    Returns
    -------
    list[str]
        Category base names such as
        ``["desenlace", "vic_grupo_social", "captura_tipo"]``.
    """
    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Taxonomy file does not exist: {display_path(taxonomy_path)}"
        )

    with open(taxonomy_path, "r", encoding="utf-8") as handle:
        taxonomy: dict[str, Any] = json.load(handle)

    label_options = taxonomy.get("label_options")
    if not isinstance(label_options, dict) or not label_options:
        raise ValueError(
            f"Taxonomy '{display_path(taxonomy_path)}' is missing a "
            "non-empty 'label_options' mapping."
        )
    return list(label_options.keys())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_input_columns(
    df: pd.DataFrame,
    category_bases: list[str],
) -> None:
    """Verify that the input frame contains the required column pairs.

    For each category base the frame must contain both the human
    annotation column (the base name itself) and the corresponding
    machine classification column (``{base}_classification``).

    Parameters
    ----------
    df : pd.DataFrame
        The loaded input frame.
    category_bases : list[str]
        Category base names from the taxonomy.

    Raises
    ------
    ValueError
        If any expected column is missing from *df*.
    """
    missing: list[str] = []
    for base in category_bases:
        if base not in df.columns:
            missing.append(base)
        cls_col = f"{base}{CLASSIFICATION_SUFFIX}"
        if cls_col not in df.columns:
            missing.append(cls_col)

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}"
        )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def add_support_flags(
    df: pd.DataFrame,
    category_bases: list[str],
) -> pd.DataFrame:
    """Add per-label support flags, an aggregate flag, and a count column.

    For each category base a boolean column ``{base}_supported`` is set
    to ``True`` when both the human annotation and the machine
    classification are non-NaN on the same row.  The
    ``any_label_supported`` column is ``True`` when at least one label
    pair on the row is supported.  The ``support_count`` column holds
    the integer number of supported label pairs per row.

    These columns are always computed internally (other toggles depend
    on them).  Whether they appear in the final output is controlled by
    the ``FLAG_SUPPORTS`` toggle.

    Parameters
    ----------
    df : pd.DataFrame
        The input frame (not modified in place).
    category_bases : list[str]
        Category base names from the taxonomy.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the added boolean and count columns.
    """
    df = df.copy()

    support_columns: list[str] = []
    for base in category_bases:
        cls_col = f"{base}{CLASSIFICATION_SUFFIX}"
        flag_col = f"{base}{SUPPORTED_SUFFIX}"

        human_present = df[base].notna()
        machine_present = df[cls_col].notna()
        df[flag_col] = human_present & machine_present

        support_columns.append(flag_col)

    df[ANY_SUPPORTED_COLUMN] = df[support_columns].any(axis=1)
    df[SUPPORT_COUNT_COLUMN] = df[support_columns].sum(axis=1).astype(int)
    return df


def mask_unsupported_annotations(
    df: pd.DataFrame,
    category_bases: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace human annotations with empty string where unsupported.

    For each label pair, if the human annotation is non-NaN but the
    machine classification is NaN (the machine has no prediction for
    this report), the human annotation value is set to ``""``.  This
    prevents downstream accuracy calculations from comparing a real
    human label against a missing machine value.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to modify (not modified in place).
    category_bases : list[str]
        Category base names from the taxonomy.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        A copy of *df* with masked values, and a dictionary mapping
        each category base to the number of annotations that were
        masked.
    """
    df = df.copy()

    masked_counts: dict[str, int] = {}
    for base in category_bases:
        cls_col = f"{base}{CLASSIFICATION_SUFFIX}"
        unsupported_mask = df[base].notna() & df[cls_col].isna()
        masked_counts[base] = int(unsupported_mask.sum())
        df.loc[unsupported_mask, base] = ""

    return df, masked_counts


def drop_support_columns(
    df: pd.DataFrame,
    category_bases: list[str],
) -> pd.DataFrame:
    """Remove the support flag and count columns from the frame.

    Called when ``FLAG_SUPPORTS`` is ``False`` so that the output CSV
    does not contain the internally-computed support metadata.

    Parameters
    ----------
    df : pd.DataFrame
        The frame to strip (not modified in place).
    category_bases : list[str]
        Category base names from the taxonomy.

    Returns
    -------
    pd.DataFrame
        A copy of *df* without the support columns.
    """
    columns_to_drop = [f"{base}{SUPPORTED_SUFFIX}" for base in category_bases]
    columns_to_drop.append(ANY_SUPPORTED_COLUMN)
    columns_to_drop.append(SUPPORT_COUNT_COLUMN)

    present = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=present).copy()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(
    df_flagged: pd.DataFrame,
    df_output: pd.DataFrame,
    category_bases: list[str],
    masked_counts: dict[str, int] | None,
) -> dict[str, Any]:
    """Build a structured summary of the support-filtering operation.

    Reports per-label counts (supported, unsupported, machine-only,
    neither) computed on the pre-filter frame, row-level totals for
    both input and output, active toggles, and masking counts when
    applicable.

    Parameters
    ----------
    df_flagged : pd.DataFrame
        The full input frame after support flags were added but before
        row filtering or column dropping.
    df_output : pd.DataFrame
        The final output frame as written to disk.
    category_bases : list[str]
        Category base names from the taxonomy.
    masked_counts : dict[str, int] | None
        Per-label count of annotations masked, or ``None`` when
        ``MASK_UNSUPPORTED`` is ``False``.

    Returns
    -------
    dict[str, Any]
        Summary dictionary with ``"per_label"`` breakdowns,
        ``"row_totals"``, ``"toggles"``, and optionally
        ``"masked_counts"``.
    """
    per_label: dict[str, dict[str, int]] = {}
    for base in category_bases:
        cls_col = f"{base}{CLASSIFICATION_SUFFIX}"
        human_present = df_flagged[base].notna()
        machine_present = df_flagged[cls_col].notna()

        per_label[base] = {
            "supported": int((human_present & machine_present).sum()),
            "unsupported": int((human_present & ~machine_present).sum()),
            "machine_only": int((~human_present & machine_present).sum()),
            "neither": int((~human_present & ~machine_present).sum()),
        }

    summary: dict[str, Any] = {
        "per_label": per_label,
        "row_totals": {
            "input_rows": len(df_flagged),
            "output_rows": len(df_output),
            "dropped_rows": len(df_flagged) - len(df_output),
        },
        "input_columns": len(df_flagged.columns),
        "output_columns": len(df_output.columns),
        "category_bases": category_bases,
        "toggles": {
            "keep_only_supported_rows": KEEP_ONLY_SUPPORTED_ROWS,
            "flag_supports": FLAG_SUPPORTS,
            "mask_unsupported": MASK_UNSUPPORTED,
        },
    }

    if masked_counts is not None:
        summary["masked_counts"] = masked_counts

    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def select_supported_rows(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    taxonomy_path: Path = TAXONOMY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the postprocessed CSV, apply toggles, and write output.

    The processing pipeline runs in this order:

    1. Load input and add support flags (always, needed internally).
    2. Mask unsupported human annotations (if ``MASK_UNSUPPORTED``).
    3. Filter to supported rows (if ``KEEP_ONLY_SUPPORTED_ROWS``).
    4. Drop support columns from output (if not ``FLAG_SUPPORTS``).
    5. Write CSV.

    Parameters
    ----------
    input_path : Path
        Path to the postprocessed classification CSV.
    output_path : Path
        Destination path for the output CSV.
    taxonomy_path : Path
        Path to the taxonomy JSON that defines category base names.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        The output frame and a structured summary dictionary.
    """
    category_bases = get_category_bases(taxonomy_path)
    df_input = load_input_frame(input_path)
    validate_input_columns(df_input, category_bases)

    df_flagged = add_support_flags(df_input, category_bases)

    masked_counts: dict[str, int] | None = None
    if MASK_UNSUPPORTED:
        df_flagged, masked_counts = mask_unsupported_annotations(
            df_flagged, category_bases,
        )

    if KEEP_ONLY_SUPPORTED_ROWS:
        df_output = df_flagged[df_flagged[ANY_SUPPORTED_COLUMN]].copy()
    else:
        df_output = df_flagged.copy()

    if not FLAG_SUPPORTS:
        df_output = drop_support_columns(df_output, category_bases)

    summary = build_summary(
        df_flagged=df_flagged,
        df_output=df_output,
        category_bases=category_bases,
        masked_counts=masked_counts,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    return df_output, summary


def main() -> int:
    """Entry point: run support selection and print a human-readable summary."""
    output_df, summary = select_supported_rows()

    print("Feature toggles:")
    for toggle_name, toggle_value in summary["toggles"].items():
        print(f"  {toggle_name}: {toggle_value}")
    print()

    row_totals = summary["row_totals"]
    print(
        f"Input rows:   {row_totals['input_rows']}\n"
        f"Output rows:  {row_totals['output_rows']}\n"
        f"Dropped rows: {row_totals['dropped_rows']}"
    )
    print(
        f"Columns: {summary['input_columns']} -> "
        f"{summary['output_columns']}"
    )
    print()

    print("Per-label support breakdown (on full input before filtering):")
    for base in summary["category_bases"]:
        counts = summary["per_label"][base]
        print(f"  {base}:")
        print(f"    supported:    {counts['supported']}")
        print(f"    unsupported:  {counts['unsupported']}")
        print(f"    machine_only: {counts['machine_only']}")
        print(f"    neither:      {counts['neither']}")
    print()

    if "masked_counts" in summary:
        print("Masked unsupported human annotations:")
        for base, count in summary["masked_counts"].items():
            print(f"  {base}: {count}")
        print()

    print(
        f"Wrote {len(output_df)} rows and {len(output_df.columns)} columns "
        f"to {display_path(OUTPUT_PATH)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
