import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.processors import MessyTextConversationState, ProcessorResult


def serialize_result_entry(
    result: ProcessorResult,
    victim_id: str,
    model_name: str,
    turn_index: int,
) -> Dict[str, Any]:
    """
    Convert a ProcessorResult into a flat, JSON-serializable dictionary.

    Args:
        result: Structured result returned by the processor.
        victim_id: Logical victim identifier.
        model_name: Name of the model that produced this result.
        turn_index: Zero-based turn index within the conversation.

    Returns:
        Dict with core fields and a JSON string for the values payload.
    """
    return {
        "victim_id": victim_id,
        "model": model_name,
        "turn_index": turn_index,
        "task_name": result.task_name,
        "doc_id": result.doc_id,
        "values": result.values,
        "error": result.error or "",
    }


def serialize_state_entry(
    state: MessyTextConversationState,
    victim_id: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Convert a MessyTextConversationState into a single record for storage.

    Args:
        state: Conversation state holding all ProcessorResult objects.
        victim_id: Logical victim identifier.
        model_name: Name of the model used during processing.

    Returns:
        Dict with victim/model identifiers, turn count, and serialized results.
    """
    serialized_results: List[Dict[str, Any]] = []
    for turn_index, result in enumerate(state.results):
        serialized_results.append(
            serialize_result_entry(
                result=result,
                victim_id=victim_id,
                model_name=model_name,
                turn_index=turn_index,
            )
        )

    return {
        "victim_id": victim_id,
        "model": model_name,
        "turns": state.turn_index,
        "results": json.dumps(serialized_results, ensure_ascii=False),
    }


def flatten_spans_from_state(
    state: MessyTextConversationState,
    victim_id: str,
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    Flatten summary_by_item spans from a conversation state into row records.

    Args:
        state: Conversation state containing ProcessorResult objects.
        victim_id: Logical victim identifier.
        model_name: Name of the model used during processing.

    Returns:
        List of dict rows suitable for CSV export, one row per span.
    """
    rows: List[Dict[str, Any]] = []

    for turn_index, result in enumerate(state.results):
        spans_dict = result.get("spans_by_item") or result.get("summary_by_item")
        if not isinstance(spans_dict, dict):
            continue

        for label_key, spans in spans_dict.items():
            if not isinstance(spans, list):
                continue
            for item in spans:
                if not isinstance(item, dict):
                    continue
                span_text = item.get("span")
                if not span_text:
                    continue
                rows.append(
                    {
                        "victim_id": victim_id,
                        "model": model_name,
                        "label_key": label_key,
                        "span": span_text,
                        # Use the runner-provided document identifier to keep
                        # spans aligned with the input index column.
                        "doc_id": result.doc_id,
                        "offset": item.get("offset", -1),
                        "turn_index": turn_index,
                        "index": result.doc_id,
                    }
                )
    return rows


def _write_csv_with_extend(
    rows: Iterable[Dict[str, Any]],
    path: Path,
    extend: bool,
    model_name: str,
) -> None:
    """
    Write rows to CSV, optionally extending existing content while replacing
    rows for the current model.
    """
    new_df = _expand_values(pd.DataFrame(list(rows)))
    _SURROGATE_RE = re.compile(r'[\ud800-\udfff]')
    for col in new_df.select_dtypes(include="object").columns:
        new_df[col] = new_df[col].apply(
            lambda x: _SURROGATE_RE.sub('', x) if isinstance(x, str) else x
        )
    if extend and path.exists():
        existing_df = pd.read_csv(path, encoding="utf-8")
        existing_df = _expand_values(existing_df)
        if "model" in existing_df.columns:
            existing_df = existing_df[existing_df["model"] != model_name]

        all_columns = sorted(set(existing_df.columns).union(new_df.columns))
        existing_df = existing_df.reindex(columns=all_columns)
        new_df = new_df.reindex(columns=all_columns)

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(path, index=False, encoding="utf-8")
        return

    new_df.to_csv(path, index=False, encoding="utf-8")


def write_results(
    rows: Iterable[Dict[str, Any]],
    path: Path,
    extend: bool,
    model_name: str,
) -> None:
    """Persist flattened result rows."""
    _write_csv_with_extend(rows, path, extend, model_name)


def _expand_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a 'values' column (dict or JSON string) into individual columns.

    If the column is absent, the DataFrame is returned unchanged.
    """
    if "values" not in df.columns:
        return df

    parsed: List[Dict[str, Any]] = []
    for val in df["values"].tolist():
        if isinstance(val, dict):
            parsed.append(val)
        elif isinstance(val, str):
            try:
                parsed.append(json.loads(val))
            except Exception:
                parsed.append({})
        else:
            parsed.append({})

    values_df = pd.json_normalize(parsed)
    values_df = values_df.rename(columns=lambda c: str(c))

    base_df = df.drop(columns=["values"]).reset_index(drop=True)
    values_df = values_df.reset_index(drop=True)
    expanded = pd.concat([base_df, values_df], axis=1)
    return expanded


def write_states(
    rows: Iterable[Dict[str, Any]],
    path: Path,
    extend: bool,
    model_name: str,
) -> None:
    """Persist serialized conversation states."""
    _write_csv_with_extend(rows, path, extend, model_name)


def write_spans(
    rows: Iterable[Dict[str, Any]],
    path: Path,
    extend: bool,
    model_name: str,
) -> None:
    """Persist flattened spans."""
    _write_csv_with_extend(rows, path, extend, model_name)
