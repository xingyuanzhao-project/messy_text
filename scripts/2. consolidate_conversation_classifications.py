"""
Consolidate conversation classification CSV exports into one CSV.

This script reads the configured classification CSV files, drops rows where
`model_classification` is missing, preserves the remaining rows, and writes
one consolidated CSV to a configurable output path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

# Update these paths here when the input/output files change.
INPUT_PATHS = [
    ROOT / "results_down_sized" / "df_text_by_report_conversation_classification (4).csv",
    ROOT
    / "results_down_sized"
    / "df_text_by_report_conversation_classification_consolidated.csv",
]
OUTPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated.csv"
)

FILTER_COLUMN = "model_classification"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_classification_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {display_path(path)}")

    df = pd.read_csv(path, encoding="utf-8", dtype=object)
    if FILTER_COLUMN not in df.columns:
        raise ValueError(
            f"Input file '{display_path(path)}' is missing required column "
            f"'{FILTER_COLUMN}'."
        )
    return df


def keep_classified_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[FILTER_COLUMN].notna()].copy()


def merge_column_order(
    current_columns: Sequence[str],
    incoming_columns: Sequence[str],
) -> list[str]:
    merged = list(current_columns)
    seen = set(merged)
    for column in incoming_columns:
        if column not in seen:
            merged.append(column)
            seen.add(column)
    return merged


def main() -> int:
    output_path = OUTPUT_PATH.resolve()
    input_paths = [path.resolve() for path in INPUT_PATHS]

    if output_path in input_paths:
        raise ValueError("OUTPUT_PATH must be different from all INPUT_PATHS.")

    frames: list[pd.DataFrame] = []
    column_order: list[str] = []
    total_rows_written = 0

    for path in input_paths:
        df = load_classification_frame(path)
        rows_before = len(df)
        df = keep_classified_rows(df)

        frames.append(df)
        column_order = merge_column_order(column_order, list(df.columns))
        total_rows_written += len(df)

        print(
            f"- {display_path(path)}: kept {len(df)} of {rows_before} rows "
            f"after dropping NaN {FILTER_COLUMN}"
        )

    combined_df = pd.concat(
        [frame.reindex(columns=column_order) for frame in frames],
        ignore_index=True,
    )

    if len(combined_df) != total_rows_written:
        raise ValueError(
            "Concatenation check failed: expected "
            f"{total_rows_written} rows, got {len(combined_df)} rows."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print(
        f"Wrote {len(combined_df)} rows and {len(combined_df.columns)} columns "
        f"to {display_path(output_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
