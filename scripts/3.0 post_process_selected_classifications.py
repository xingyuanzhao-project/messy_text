"""
Post-process selected label columns from the consolidated classification CSV.

This script keeps all non-category columns unchanged, retains only the selected
category bases defined in config/taxonomy.json and their `_classification`
partners, drops the other 12 category base columns from config/taxonomy_15.json,
adds `turn_index` from the per-model records exports, collapses selected labels
using config/taxonomy.json mappings, and blanks fallback values after
collapsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated.csv"
)
OUTPUT_PATH = (
    ROOT
    / "results_down_sized"
    / "df_text_by_report_classification_consolidated_eval.csv"
)
SELECTED_TAXONOMY_PATH = ROOT / "config" / "taxonomy.json"
FULL_TAXONOMY_PATH = ROOT / "config" / "taxonomy_15.json"
RECORDS_ROOT = ROOT / "results_down_sized"

CLASSIFICATION_SUFFIX = "_classification"
EXPECTED_TOTAL_CATEGORY_COUNT = 15
EXPECTED_SELECTED_CATEGORY_COUNT = 3
EXPECTED_DROPPED_CATEGORY_COUNT = 12
EXPECTED_RECORDS_FILE_COUNT = 3
RECORDS_FILE_NAME = "df_records_results.csv"
RECORDS_REQUIRED_COLUMNS = ["model", "doc_id", "turn_index", "task_name"]
TARGET_TASK_NAME = "conversation_summary"

# ── Feature toggles ──────────────────────────────────────────────────────────
DROP_UNSELECTED_COLUMNS = True
ADD_TURN_INDEX = True
COLLAPSE_LABELS = True
BLANK_FALLBACK = True


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {display_path(path)}")

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_input_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {display_path(path)}")
    return pd.read_csv(path, encoding="utf-8", dtype=object)


def discover_records_paths(records_root: Path) -> list[Path]:
    records_paths = sorted(records_root.glob(f"*/{RECORDS_FILE_NAME}"))
    if len(records_paths) != EXPECTED_RECORDS_FILE_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_RECORDS_FILE_COUNT} records files named "
            f"'{RECORDS_FILE_NAME}' under {display_path(records_root)}, got "
            f"{len(records_paths)}."
        )
    return records_paths


def load_records_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Records file does not exist: {display_path(path)}")

    return pd.read_csv(
        path,
        encoding="utf-8",
        dtype=object,
        usecols=RECORDS_REQUIRED_COLUMNS,
    )


def build_turn_index_lookup(
    records_root: Path,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    records_paths = discover_records_paths(records_root)
    records_frames: list[pd.DataFrame] = []
    model_directory_map: dict[str, str] = {}

    for path in records_paths:
        records_df = load_records_frame(path)
        records_df = records_df[
            records_df["task_name"].astype(str).str.strip() == TARGET_TASK_NAME
        ].copy()

        unique_models = sorted(
            {
                str(model).strip()
                for model in records_df["model"].dropna()
                if str(model).strip()
            }
        )
        if not unique_models:
            raise ValueError(
                f"No non-empty model values found in {display_path(path)} after "
                f"filtering task_name == '{TARGET_TASK_NAME}'."
            )

        directory_display = display_path(path.parent)
        for model_name in unique_models:
            previous_directory = model_directory_map.get(model_name)
            if previous_directory is not None and previous_directory != directory_display:
                raise ValueError(
                    f"Model '{model_name}' appears in multiple records "
                    f"directories: {previous_directory} and {directory_display}"
                )
            model_directory_map[model_name] = directory_display

        records_frames.append(records_df)

    lookup_df = pd.concat(records_frames, ignore_index=True)
    duplicate_mask = lookup_df.duplicated(subset=["model", "doc_id"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = (
            lookup_df.loc[duplicate_mask, ["model", "doc_id"]]
            .drop_duplicates()
            .sort_values(["model", "doc_id"])
        )
        duplicate_preview = duplicate_pairs.head(10).to_dict("records")
        raise ValueError(
            "Duplicate (model, doc_id) keys found in the concatenated records "
            f"lookup: {duplicate_preview}"
        )

    lookup_df = lookup_df.rename(columns={"doc_id": "index"})
    lookup_df["turn_index"] = pd.to_numeric(
        lookup_df["turn_index"],
        errors="coerce",
    ).astype("Int64")

    records_files_loaded = [display_path(path) for path in records_paths]
    return lookup_df[["model", "index", "turn_index"]], model_directory_map, records_files_loaded


def get_category_bases(taxonomy: dict[str, Any], taxonomy_name: str) -> list[str]:
    label_options = taxonomy.get("label_options")
    if not isinstance(label_options, dict) or not label_options:
        raise ValueError(
            f"Taxonomy '{taxonomy_name}' is missing a non-empty 'label_options' mapping."
        )
    return list(label_options.keys())


def validate_category_sets(
    full_category_bases: list[str],
    selected_category_bases: list[str],
) -> list[str]:
    full_category_set = set(full_category_bases)
    selected_category_set = set(selected_category_bases)

    if len(full_category_set) != EXPECTED_TOTAL_CATEGORY_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_TOTAL_CATEGORY_COUNT} total category bases in "
            f"{display_path(FULL_TAXONOMY_PATH)}, got {len(full_category_set)}."
        )

    if len(selected_category_set) != EXPECTED_SELECTED_CATEGORY_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_SELECTED_CATEGORY_COUNT} selected category bases in "
            f"{display_path(SELECTED_TAXONOMY_PATH)}, got {len(selected_category_set)}."
        )

    if not selected_category_set.issubset(full_category_set):
        unexpected = sorted(selected_category_set - full_category_set)
        raise ValueError(
            "Selected taxonomy contains category bases not present in the full "
            f"15-category codebook: {unexpected}"
        )

    dropped_category_bases = sorted(full_category_set - selected_category_set)
    if len(dropped_category_bases) != EXPECTED_DROPPED_CATEGORY_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_DROPPED_CATEGORY_COUNT} unselected category bases, got "
            f"{len(dropped_category_bases)}."
        )

    return dropped_category_bases


def validate_input_schema(
    df: pd.DataFrame,
    full_category_bases: list[str],
    selected_category_bases: list[str],
    available_record_models: set[str] | None,
) -> None:
    full_category_set = set(full_category_bases)
    selected_classification_columns = {
        f"{base}{CLASSIFICATION_SUFFIX}" for base in selected_category_bases
    }

    missing_category_bases = sorted(
        base for base in full_category_bases if base not in df.columns
    )
    if missing_category_bases:
        raise ValueError(
            "Input CSV is missing expected category base columns from "
            f"{display_path(FULL_TAXONOMY_PATH)}: {missing_category_bases}"
        )

    missing_selected_classification_columns = sorted(
        col for col in selected_classification_columns if col not in df.columns
    )
    if missing_selected_classification_columns:
        raise ValueError(
            "Input CSV is missing expected selected classification columns: "
            f"{missing_selected_classification_columns}"
        )

    unexpected_category_classification_columns = sorted(
        col
        for col in df.columns
        if col.endswith(CLASSIFICATION_SUFFIX)
        and col[: -len(CLASSIFICATION_SUFFIX)] in full_category_set
        and col not in selected_classification_columns
    )
    if unexpected_category_classification_columns:
        raise ValueError(
            "Input CSV contains category classification columns outside the "
            "selected scope: "
            f"{unexpected_category_classification_columns}"
        )

    if available_record_models is not None:
        input_models = {
            str(model).strip() for model in df["model"].dropna() if str(model).strip()
        }
        missing_record_models = sorted(input_models - available_record_models)
        if missing_record_models:
            raise ValueError(
                "Input CSV contains model values that do not exist in the "
                f"discovered records files: {missing_record_models}"
            )


def post_process_value(
    value: object,
    mapping: dict[str, str],
    fallback_values_lower: set[str],
    collapse: bool,
    blank_fallback: bool,
) -> str:
    if pd.isna(value):
        return ""

    value_str = str(value).strip()
    if not value_str:
        return ""

    if collapse:
        value_str = mapping.get(value_str, value_str).strip()
        if not value_str:
            return ""

    if blank_fallback and value_str.lower() in fallback_values_lower:
        return ""

    return value_str


def build_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    dropped_category_columns: list[str],
    selected_category_bases: list[str],
    records_files_loaded: list[str],
    model_directory_map: dict[str, str],
    unmatched_turn_index_rows: int,
) -> dict[str, Any]:
    selected_classification_columns = [
        f"{base}{CLASSIFICATION_SUFFIX}" for base in selected_category_bases
    ]
    return {
        "rows": len(df_after),
        "original_column_count": len(df_before.columns),
        "final_column_count": len(df_after.columns),
        "dropped_category_columns": dropped_category_columns,
        "selected_category_columns": selected_category_bases,
        "selected_classification_columns": selected_classification_columns,
        "records_files_loaded": records_files_loaded,
        "model_directory_map": model_directory_map,
        "unmatched_turn_index_rows": unmatched_turn_index_rows,
    }


def post_process_selected_classifications(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    selected_taxonomy_path: Path = SELECTED_TAXONOMY_PATH,
    full_taxonomy_path: Path = FULL_TAXONOMY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_taxonomy = load_json(selected_taxonomy_path)
    full_taxonomy = load_json(full_taxonomy_path)
    df_input = load_input_frame(input_path)

    if ADD_TURN_INDEX:
        turn_index_lookup, model_directory_map, records_files_loaded = (
            build_turn_index_lookup(RECORDS_ROOT)
        )
    else:
        turn_index_lookup = None
        model_directory_map = {}
        records_files_loaded = []

    selected_category_bases = get_category_bases(
        selected_taxonomy,
        display_path(selected_taxonomy_path),
    )
    full_category_bases = get_category_bases(
        full_taxonomy,
        display_path(full_taxonomy_path),
    )

    dropped_category_columns = validate_category_sets(
        full_category_bases=full_category_bases,
        selected_category_bases=selected_category_bases,
    )
    validate_input_schema(
        df=df_input,
        full_category_bases=full_category_bases,
        selected_category_bases=selected_category_bases,
        available_record_models=(
            set(model_directory_map) if ADD_TURN_INDEX else None
        ),
    )

    if ADD_TURN_INDEX:
        df_enriched = df_input.merge(
            turn_index_lookup,
            how="left",
            on=["model", "index"],
            validate="many_to_one",
        )
        df_enriched["turn_index"] = pd.to_numeric(
            df_enriched["turn_index"],
            errors="coerce",
        ).astype("Int64")
        unmatched_turn_index_rows = int(df_enriched["turn_index"].isna().sum())
    else:
        df_enriched = df_input.copy()
        unmatched_turn_index_rows = 0

    if DROP_UNSELECTED_COLUMNS:
        columns_to_drop = list(dropped_category_columns) + [
            f"{base}{CLASSIFICATION_SUFFIX}"
            for base in dropped_category_columns
            if f"{base}{CLASSIFICATION_SUFFIX}" in df_enriched.columns
        ]
        df_output = df_enriched.drop(
            columns=columns_to_drop, errors="raise",
        ).copy()
    else:
        df_output = df_enriched.copy()

    if COLLAPSE_LABELS or BLANK_FALLBACK:
        category_merging = selected_taxonomy.get("category_merging", {})
        fallback_values = selected_taxonomy.get("fallback_values", [])
        fallback_values_lower = {
            str(value).strip().lower()
            for value in fallback_values
            if str(value).strip()
        }

        for base in selected_category_bases:
            mapping = category_merging.get(base, {})
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"category_merging for '{base}' must be a mapping in "
                    f"{display_path(selected_taxonomy_path)}."
                )

            columns_to_process = [base, f"{base}{CLASSIFICATION_SUFFIX}"]
            for column in columns_to_process:
                df_output[column] = df_output[column].apply(
                    lambda value, current_mapping=mapping: post_process_value(
                        value=value,
                        mapping=current_mapping,
                        fallback_values_lower=fallback_values_lower,
                        collapse=COLLAPSE_LABELS,
                        blank_fallback=BLANK_FALLBACK,
                    )
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    summary = build_summary(
        df_before=df_input,
        df_after=df_output,
        dropped_category_columns=(
            dropped_category_columns if DROP_UNSELECTED_COLUMNS else []
        ),
        selected_category_bases=selected_category_bases,
        records_files_loaded=records_files_loaded,
        model_directory_map=model_directory_map,
        unmatched_turn_index_rows=unmatched_turn_index_rows,
    )
    summary["toggles"] = {
        "drop_unselected_columns": DROP_UNSELECTED_COLUMNS,
        "add_turn_index": ADD_TURN_INDEX,
        "collapse_labels": COLLAPSE_LABELS,
        "blank_fallback": BLANK_FALLBACK,
    }
    return df_output, summary


def main() -> int:
    output_df, summary = post_process_selected_classifications()

    print("Feature toggles:")
    for toggle_name, toggle_value in summary["toggles"].items():
        print(f"  {toggle_name}: {toggle_value}")
    print(f"Rows: {summary['rows']}")
    print(
        "Columns: "
        f"{summary['original_column_count']} -> {summary['final_column_count']}"
    )
    print(
        "Selected category columns kept: "
        + ", ".join(summary["selected_category_columns"])
    )
    print(
        "Selected classification columns kept: "
        + ", ".join(summary["selected_classification_columns"])
    )
    print(
        "Records files loaded "
        f"({len(summary['records_files_loaded'])}): "
        + ", ".join(summary["records_files_loaded"])
    )
    print("Model to records directory map:")
    for model_name, directory in sorted(summary["model_directory_map"].items()):
        print(f"  {model_name} -> {directory}")
    print(f"Rows with unmatched turn_index: {summary['unmatched_turn_index_rows']}")
    print(
        "Dropped category columns "
        f"({len(summary['dropped_category_columns'])}): "
        + ", ".join(summary["dropped_category_columns"])
    )
    print(
        "Wrote "
        f"{len(output_df)} rows and {len(output_df.columns)} columns to "
        f"{display_path(OUTPUT_PATH)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
