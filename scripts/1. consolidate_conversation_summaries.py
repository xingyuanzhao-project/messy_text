"""
Consolidate per-model conversation summary CSV exports into one long-format CSV.

This script discovers summary exports under ``results_down_sized``, validates
the expected schema contract, preserves the full row content, and writes a
single consolidated CSV to a non-input filename.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = ROOT / "results_down_sized"
DEFAULT_PATTERN = "df_text_by_report_conversation_summary_less_label.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_conversation_summary_less_label_consolidated.csv"
)
REQUIRED_COLUMNS = (
    "index",
    "victim",
    "text",
    "summary_all_context",
    "model",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate per-model conversation summary CSV exports into a "
            "single long-format CSV without dropping rows."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory to scan for summary CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Pattern passed to Path.rglob() when discovering inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV path for the consolidated output.",
    )
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_input_paths(
    input_root: Path,
    pattern: str,
    output_path: Path,
) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {display_path(input_root)}"
        )

    candidates = sorted(
        path.resolve()
        for path in input_root.rglob(pattern)
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No input files found for pattern "
            f"'{pattern}' under {display_path(input_root)}"
        )

    input_paths = [path for path in candidates if path != output_path]
    if not input_paths:
        raise ValueError(
            "All discovered files resolve to the output path. "
            "Choose a different output filename."
        )
    return input_paths


def load_summary_frame(path: Path) -> pd.DataFrame:
    # Load as object data and keep empty strings intact so the script does not
    # coerce or drop literal cell content while consolidating raw exports.
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype=object,
        keep_default_na=False,
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Input file '{display_path(path)}' is missing required columns: "
            f"{missing}"
        )
    return df


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


def duplicate_key_stats(df: pd.DataFrame) -> tuple[int, int]:
    duplicate_mask = df.duplicated(subset=["model", "index"], keep=False)
    if not duplicate_mask.any():
        return 0, 0

    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(
        df.loc[duplicate_mask, ["model", "index"]].drop_duplicates().shape[0]
    )
    return duplicate_groups, duplicate_rows


def verify_round_trip(
    output_path: Path,
    expected_rows: int,
    expected_columns: Sequence[str],
) -> None:
    verified_df = load_summary_frame(output_path)
    if len(verified_df) != expected_rows:
        raise ValueError(
            "Round-trip validation failed: expected "
            f"{expected_rows} rows, found {len(verified_df)} rows in "
            f"{display_path(output_path)}"
        )
    if list(verified_df.columns) != list(expected_columns):
        raise ValueError(
            "Round-trip validation failed: output columns changed during write."
        )


def main() -> int:
    args = parse_args()
    input_root = resolve_workspace_path(args.input_root)
    output_path = resolve_workspace_path(args.output)

    try:
        input_paths = discover_input_paths(
            input_root=input_root,
            pattern=args.pattern,
            output_path=output_path,
        )

        print(
            f"Found {len(input_paths)} input file(s) matching '{args.pattern}' "
            f"under {display_path(input_root)}"
        )

        frames: list[pd.DataFrame] = []
        column_order: list[str] = []
        total_input_rows = 0

        for path in input_paths:
            df = load_summary_frame(path)
            frames.append(df)
            column_order = merge_column_order(column_order, list(df.columns))
            total_input_rows += len(df)
            print(
                f"- {display_path(path)}: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )

        aligned_frames = [
            frame.reindex(columns=column_order)
            for frame in frames
        ]
        combined_df = pd.concat(aligned_frames, ignore_index=True)

        if len(combined_df) != total_input_rows:
            raise ValueError(
                "Concatenation check failed: expected "
                f"{total_input_rows} rows, got {len(combined_df)} rows."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )

        duplicate_groups, duplicate_rows = duplicate_key_stats(combined_df)

        print(
            f"Wrote {len(combined_df)} rows and {len(combined_df.columns)} "
            f"columns to {display_path(output_path)}"
        )
        if duplicate_groups:
            print(
                "WARNING: Found "
                f"{duplicate_groups} duplicate (model, index) key(s) spanning "
                f"{duplicate_rows} row(s). Rows were preserved unchanged."
            )
        else:
            print("Duplicate check: no duplicate (model, index) keys found.")

        verify_round_trip(
            output_path=output_path,
            expected_rows=len(combined_df),
            expected_columns=column_order,
        )
        print(
            "Round-trip check passed: output row count and columns reload "
            "cleanly with the CSV parser."
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
