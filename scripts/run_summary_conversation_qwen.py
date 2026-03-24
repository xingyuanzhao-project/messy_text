"""
Conversation-based multi-turn processing entry point for MessyText.

This script implements a multi-turn, victim-level summarization pipeline:

- Input is expected to be a report-level CSV (e.g. df_text_by_report.csv)
  with at least the columns:
    - 'index'  : document identifier (do not confuse with pandas index)
    - 'victim' : victim identifier used to group documents
    - 'text'   : raw document text

- For each victim, all associated documents are processed sequentially:
    1. The first document is summarized.
    2. The summary is updated turn-by-turn as additional documents are seen.
    3. The updated summary is attached back to each document row so that
       downstream components can decide whether to use the per-turn or
       final (last-turn) summary.

- Victims can be processed concurrently using asyncio to fully utilize
  high-throughput backends such as vLLM.

Configuration
-------------

The script reads settings from the file path stored in the module-level
`settings` variable (default: config/settings.yaml) and respects the
following keys:

    model:
      name: ...
      api_base: ...
      api_key: ...

    paths:
      input: "df_text_by_report.csv"
      output:
        file: "df_text_multi_eval_by_victim.csv"
        extend: false
      taxonomy: "config/taxonomy.json"

    async:
      enabled: true
      max_retries: 5
      max_concurrent_rows: 5   # used here as max_concurrent_victims

    processing:
      temperature: 0.0
      max_tokens_summary: 1024
      max_tokens_classification: 256
      conversation:
        enabled: false              # opt-in for this script

    display:
      use_progress_bar: true

    logging:
      file: "processing.log"
      log_progress: true

When processing.conversation.enabled is false, this script will exit
without modifying data, so existing pipelines using run_processing.py
remain unchanged.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as tqdm_async
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set the YAML file to load before running this script.
settings = "config/settings_qwen.yaml"

from src.processors import (  # noqa: E402
    AsyncMessyTextConversationOrchestrator,
    AsyncMessyTextConversationTurnProcessor,
    AsyncMessyTextProcessor,
    MessyTextConversationOrchestrator,
    MessyTextConversationState,
    MessyTextConversationTurnProcessor,
    MessyTextProcessor,
)
from src.recorders import (  # noqa: E402
    flatten_spans_from_state,
    serialize_result_entry,
    serialize_state_entry,
    write_results,
    write_spans,
    write_states,
)
from src.utils import (  # noqa: E402
    check_gpu_info,
    check_vllm_server,
    is_informative_summary,
    setup_logger,
)


async def _process_victim_async(
    victim_id: str,
    group_df: pd.DataFrame,
    turn_processor: AsyncMessyTextConversationTurnProcessor,
) -> Tuple[str, Dict[int, str], MessyTextConversationState]:
    """
    Asynchronously processes all documents for a single victim.

    Args:
        victim_id (str): Identifier for the victim (from the 'victim' column).
        group_df (pd.DataFrame): Subset of the input DataFrame for this victim.
        turn_processor (AsyncMessyTextConversationTurnProcessor): Per-turn
            processor used to obtain candidate summaries from the model.

    Returns:
        Tuple[str, Dict[int, str], MessyTextConversationState]:
            - The victim_id.
            - A mapping from pandas row index to the per-turn summary string.
            - The final conversation state containing all turn results.
    """

    # Ensure deterministic ordering of documents within a victim.
    # The 'index' column is the document identifier, distinct from the
    # DataFrame's own index.
    group_sorted = group_df.sort_values(by="index")
    index_list = group_sorted.index.tolist()
    texts: List[str] = [str(t) for t in group_sorted["text"]]
    doc_ids = list(group_sorted["index"])

    # Runner-owned conversation memory. This tracks the running summary and
    # ensures that non-informative documents do not overwrite existing context.
    running_summary: str = ""
    state = MessyTextConversationState(turn_index=0)
    per_row_summaries: List[str] = []

    for doc_id, raw_text in zip(doc_ids, texts):
        # Call the per-turn processor with the current running summary as the
        # previous_summary. We ignore the returned state's last_summary in
        # favor of the runner-owned running_summary, but we do use the state's
        # last_result to decide whether this turn is informative.
        candidate_summary, turn_state = await turn_processor.process_turn(
            raw_text=raw_text,
            state=state,
            doc_id=doc_id,
        )

        result = getattr(turn_state, "last_result", None)

        has_info = False
        if result is not None and hasattr(result, "has_field") and result.has_field("info_found"):
            flag = str(result.get("info_found") or "").strip().lower()
            has_info = flag not in {"", "false", "0", "no"}
        elif result is not None and hasattr(result, "is_no_info"):
            has_info = not result.is_no_info()
        else:
            # Fallback to string-based heuristic if structured result is
            # unavailable for any reason.
            has_info = is_informative_summary(candidate_summary)

        if has_info:
            # Prefer the structured summary field when available.
            new_summary = result.get("summary") if result is not None else candidate_summary
            running_summary = (new_summary or "").strip()

        per_row_summaries.append(running_summary)

        state = turn_state

    # Map per-row summaries back to the original pandas indices of group_sorted.
    index_to_summary: Dict[int, str] = {}
    for row_idx, summary in zip(index_list, per_row_summaries):
        index_to_summary[row_idx] = summary

    return victim_id, index_to_summary, state


def _process_victim_sync(
    victim_id: str,
    group_df: pd.DataFrame,
    turn_processor: MessyTextConversationTurnProcessor,
    use_progress_bar: bool,
) -> Tuple[str, Dict[int, str], MessyTextConversationState]:
    """
    Synchronously processes all documents for a single victim.

    Args:
        victim_id (str): Identifier for the victim (from the 'victim' column).
        group_df (pd.DataFrame): Subset of the input DataFrame for this victim.
        turn_processor (MessyTextConversationTurnProcessor): Per-turn processor
            used to obtain candidate summaries from the model.

    Returns:
        Tuple[str, Dict[int, str], MessyTextConversationState]:
            - The victim_id.
            - A mapping from pandas row index to the per-turn summary string.
            - The final conversation state containing all turn results.
    """

    group_sorted = group_df.sort_values(by="index")
    index_list = group_sorted.index.tolist()
    texts: List[str] = [str(t) for t in group_sorted["text"]]
    doc_ids = list(group_sorted["index"])

    running_summary: str = ""
    state = MessyTextConversationState(turn_index=0)
    per_row_summaries: List[str] = []

    # Optional document-level progress bar within each victim, mirroring the
    # nested tqdm usage in the classification pipeline.
    if use_progress_bar:
        with tqdm(
            total=len(texts),
            desc=f"Docs (victim={victim_id})",
            leave=False,
            position=1,
        ) as pbar_docs:
            for doc_id, raw_text in zip(doc_ids, texts):
                candidate_summary, turn_state = turn_processor.process_turn(
                    raw_text=raw_text,
                    state=state,
                    doc_id=doc_id,
                )

                result = getattr(turn_state, "last_result", None)

                has_info = False
                if result is not None and hasattr(result, "has_field") and result.has_field("info_found"):
                    flag = str(result.get("info_found") or "").strip().lower()
                    has_info = flag not in {"", "false", "0", "no"}
                elif result is not None and hasattr(result, "is_no_info"):
                    has_info = not result.is_no_info()
                else:
                    has_info = is_informative_summary(candidate_summary)

                if has_info:
                    new_summary = result.get("summary") if result is not None else candidate_summary
                    running_summary = (new_summary or "").strip()

                per_row_summaries.append(running_summary)

                state = turn_state
                pbar_docs.update(1)
    else:
        for doc_id, raw_text in zip(doc_ids, texts):
            candidate_summary, turn_state = turn_processor.process_turn(
                raw_text=raw_text,
                state=state,
                doc_id=doc_id,
            )

            result = getattr(turn_state, "last_result", None)

            has_info = False
            if result is not None and hasattr(result, "has_field") and result.has_field("info_found"):
                flag = str(result.get("info_found") or "").strip().lower()
                has_info = flag not in {"", "false", "0", "no"}
            elif result is not None and hasattr(result, "is_no_info"):
                has_info = not result.is_no_info()
            else:
                has_info = is_informative_summary(candidate_summary)

            if has_info:
                new_summary = result.get("summary") if result is not None else candidate_summary
                running_summary = (new_summary or "").strip()

            per_row_summaries.append(running_summary)

            state = turn_state

    index_to_summary: Dict[int, str] = {}
    for row_idx, summary in zip(index_list, per_row_summaries):
        index_to_summary[row_idx] = summary

    return victim_id, index_to_summary, state


async def _process_dataframe_conversation_async(
    df: pd.DataFrame,
    async_client: AsyncOpenAI,
    config: Dict,
    prompts: Dict,
    logger,
) -> Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
    """
    Asynchronously processes a report-level DataFrame using victim-level
    multi-turn conversations.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'index', 'victim',
            and 'text' columns (at minimum).
        async_client (AsyncOpenAI): Asynchronous OpenAI/vLLM client.
        config (Dict): Full configuration dictionary from settings.yaml.
        prompts (Dict): Prompt configuration loaded from config/prompts.json.
        logger: Logger instance.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'summary_all_context' column
        containing the per-turn conversation summaries.
    """
    processing_cfg = config.get("processing", {})
    async_cfg = config.get("async", {}) or {}

    max_concurrent_victims = async_cfg.get("max_concurrent_rows", 5)
    use_progress_bar = config.get("display", {}).get("use_progress_bar", True)
    summary_row_limit = processing_cfg.get("summary_row_limit")

    taxonomy_path = config["paths"]["taxonomy"]
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    processor = AsyncMessyTextProcessor(
        client=async_client,
        config={
            "model": {"name": config["model"]["name"]},
            "processing": {
                "temperature": processing_cfg["temperature"],
                "max_tokens_summary": processing_cfg["max_tokens_summary"],
                "max_tokens_classification": processing_cfg["max_tokens_classification"],
            },
            "prompts": prompts,
            "logging": config.get("logging", {}),
        },
        taxonomy=taxonomy,
        logger=logger,
    )
    turn_processor = AsyncMessyTextConversationTurnProcessor(processor)

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        df_processed["summary_all_context"] = ""

    # Group by victim and apply optional summary_row_limit to limit the number of
    # victims processed. This is analogous to the row-level summary_row_limit used in
    # the classification pipeline but applied at the victim level here.
    victim_groups = list(df_processed.groupby("victim"))
    if summary_row_limit is not None:
        victim_groups = victim_groups[:summary_row_limit]

    semaphore = asyncio.Semaphore(max_concurrent_victims)
    tasks: List[asyncio.Future] = []

    for victim_id, group_df in victim_groups:

        async def _bounded_task(v_id=victim_id, g_df=group_df):
            async with semaphore:
                return await _process_victim_async(
                    v_id,
                    g_df,
                    turn_processor,
                )

        tasks.append(_bounded_task())

    results: List[Tuple[str, Dict[int, str], MessyTextConversationState]] = []
    for coro in tqdm_async.as_completed(
        tasks,
        total=len(tasks),
        desc="Processing victims (conversation)",
        disable=not use_progress_bar,
    ):
        result = await coro
        results.append(result)

    # Apply summaries back to the DataFrame using the collected mappings.
    for _victim_id, index_to_summary, _state in results:
        for row_idx, summary in index_to_summary.items():
            df_processed.at[row_idx, "summary_all_context"] = summary

    victim_states: List[Tuple[str, MessyTextConversationState]] = [
        (victim_id, state) for victim_id, _summary_map, state in results
    ]

    return df_processed, victim_states


def _process_dataframe_conversation_sync(
    df: pd.DataFrame,
    sync_client: OpenAI,
    config: Dict,
    prompts: Dict,
    logger,
) -> Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
    """
    Synchronously processes a report-level DataFrame using victim-level
    multi-turn conversations.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'index', 'victim',
            and 'text' columns (at minimum).
        sync_client (OpenAI): Synchronous OpenAI/vLLM client.
        config (Dict): Full configuration dictionary from settings.yaml.
        prompts (Dict): Prompt configuration loaded from config/prompts.json.
        logger: Logger instance.

    Returns:
        pd.DataFrame: Updated DataFrame with a 'summary_all_context' column
        containing the per-turn conversation summaries.
    """
    processing_cfg = config.get("processing", {})

    use_progress_bar = config.get("display", {}).get("use_progress_bar", True)
    summary_row_limit = processing_cfg.get("summary_row_limit")

    taxonomy_path = config["paths"]["taxonomy"]
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    processor = MessyTextProcessor(
        client=sync_client,
        config={
            "model": {"name": config["model"]["name"]},
            "processing": {
                "temperature": processing_cfg["temperature"],
                "max_tokens_summary": processing_cfg["max_tokens_summary"],
                "max_tokens_classification": processing_cfg["max_tokens_classification"],
            },
            "prompts": prompts,
            "logging": config.get("logging", {}),
        },
        taxonomy=taxonomy,
        logger=logger,
    )
    turn_processor = MessyTextConversationTurnProcessor(processor)

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        df_processed["summary_all_context"] = ""

    victim_groups = list(df_processed.groupby("victim"))
    if summary_row_limit is not None:
        victim_groups = victim_groups[:summary_row_limit]

    results: List[Tuple[str, Dict[int, str], MessyTextConversationState]] = []
    with tqdm(
        total=len(victim_groups),
        desc="Processing victims (conversation)",
        disable=not use_progress_bar,
    ) as pbar:
        for victim_id, group_df in victim_groups:
            result = _process_victim_sync(
                victim_id,
                group_df,
                turn_processor,
                use_progress_bar=use_progress_bar,
            )
            results.append(result)
            pbar.update(1)

    for _victim_id, index_to_summary, _state in results:
        for row_idx, summary in index_to_summary.items():
            df_processed.at[row_idx, "summary_all_context"] = summary

    victim_states: List[Tuple[str, MessyTextConversationState]] = [
        (victim_id, state) for victim_id, _summary_map, state in results
    ]

    return df_processed, victim_states


def main() -> None:
    """
    Main entry point for victim-level conversation processing.

    This function:
        1. Loads configuration and input data.
        2. Performs the same pre-flight checks as run_processing.py:
           - GPU availability via nvidia-smi.
           - vLLM server connectivity and model availability.
        3. Checks whether conversation processing is enabled. If not, logs
           a message and exits without modifying data.
        4. Executes either async or sync victim-level conversation processing
           depending on the async.enabled setting.
        5. Writes the resulting DataFrame to the configured output path.
    """
    # Step 1: Load configuration
    with open(settings, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(log_file=config["logging"]["file"])
    
    # Load prompt configuration
    prompts_path = config["paths"].get("prompts", "config/prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    processing_cfg = config.get("processing", {})
    conversation_cfg = processing_cfg.get("conversation", {}) or {}
    conversation_enabled = conversation_cfg.get("enabled", False)

    if not conversation_enabled:
        logger.info(
            "Conversation processing is disabled in config (processing.conversation.enabled=false). "
            "Exiting without changes."
        )
        return

    # Step 2: Load data (conversation = summary-style processing)
    summary_paths = config["paths"]["summary"]
    input_path = summary_paths["input"]
    df_text = pd.read_csv(input_path, encoding="utf-8")

    required_columns = {"index", "victim", "text"}
    missing = required_columns - set(df_text.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns for conversation processing: {missing}"
        )

    # Step 3: Pre-flight checks (GPU + vLLM server)
    sync_client = OpenAI(
        base_url=config["model"]["api_base"],
        api_key=config["model"]["api_key"],
    )

    logger.info("=" * 50)
    logger.info("PRE-FLIGHT CHECKS (Conversation Mode)")
    logger.info("=" * 50)

    gpu_info = check_gpu_info(logger)
    if gpu_info is None:
        logger.error("No GPU available. Exiting.")
        sys.exit(1)

    vllm_ok, available_models, test_result = check_vllm_server(
        sync_client,
        config["model"]["name"],
        logger,
    )

    if not vllm_ok:
        logger.error(f"Pre-flight FAILED. Models found: {available_models}")
        sys.exit(1)

    logger.info("Pre-flight checks PASSED")
    logger.info("=" * 50)

    # Step 4: Select processing mode (async vs sync)
    async_enabled = config.get("async", {}).get("enabled", True)

    if async_enabled:
        logger.info("Using ASYNC victim-level conversation processing.")
        max_retries = config["async"].get("max_retries", 2)
        async_client = AsyncOpenAI(
            base_url=config["model"]["api_base"],
            api_key=config["model"]["api_key"],
            max_retries=max_retries,
        )
        processed_df, victim_states = asyncio.run(
            _process_dataframe_conversation_async(
                df=df_text,
                async_client=async_client,
                config=config,
                prompts=prompts,
                logger=logger,
            )
        )
    else:
        logger.info("Using SYNC victim-level conversation processing.")
        processed_df, victim_states = _process_dataframe_conversation_sync(
            df=df_text,
            sync_client=sync_client,
            config=config,
            prompts=prompts,
            logger=logger,
        )

    # Step 5: Save output
    model_name = config["model"]["name"]
    processed_df["model"] = model_name
    if "summary_all_context" in processed_df.columns:
        processed_df["summary_all_context"].replace(
            ["No information", "No relevant information found"],
            "",
            inplace=True,
        )

    summary_output_paths = config["paths"]["summary"]["output"]
    output_path = summary_output_paths["file"]
    extend_mode = summary_output_paths.get("extend", False)

    if extend_mode and Path(output_path).exists():
        existing_df = pd.read_csv(output_path, encoding="utf-8")
        rows_before = len(existing_df)
        existing_df = existing_df[existing_df["model"] != model_name]
        if len(existing_df) < rows_before:
            logger.info(
                f"Overwriting {rows_before - len(existing_df)} existing rows for model '{model_name}'"
            )
        combined_df = pd.concat([existing_df, processed_df], ignore_index=True)
        combined_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(
            f"Output extended: {len(existing_df)} existing + {len(processed_df)} new = {len(combined_df)} total rows"
        )
    else:
        processed_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Output saved to {output_path} ({len(processed_df)} rows)")

    # Step 6: Persist conversation records (results, states, spans) if configured
    records_cfg = config["paths"].get("records", {})
    if victim_states and records_cfg:
        results_rows = []
        states_rows = []
        spans_rows = []

        for victim_id, state in victim_states:
            states_rows.append(
                serialize_state_entry(
                    state=state,
                    victim_id=victim_id,
                    model_name=model_name,
                )
            )
            spans_rows.extend(
                flatten_spans_from_state(
                    state=state,
                    victim_id=victim_id,
                    model_name=model_name,
                )
            )
            for turn_index, result in enumerate(state.results):
                results_rows.append(
                    serialize_result_entry(
                        result=result,
                        victim_id=victim_id,
                        model_name=model_name,
                        turn_index=turn_index,
                    )
                )

        # Results
        results_output_cfg = (records_cfg.get("results") or {}).get("output", {})
        results_path = Path(results_output_cfg.get("file", "conversation_results.csv"))
        results_extend = results_output_cfg.get("extend", False)
        write_results(
            rows=results_rows,
            path=results_path,
            extend=results_extend,
            model_name=model_name,
        )
        logger.info(f"Results records saved to {results_path} ({len(results_rows)} rows)")

        # States
        states_output_cfg = (records_cfg.get("states") or {}).get("output", {})
        states_path = Path(states_output_cfg.get("file", "conversation_states.csv"))
        states_extend = states_output_cfg.get("extend", False)
        write_states(
            rows=states_rows,
            path=states_path,
            extend=states_extend,
            model_name=model_name,
        )
        logger.info(f"State records saved to {states_path} ({len(states_rows)} rows)")

        # Spans
        spans_output_cfg = (records_cfg.get("spans") or {}).get("output", {})
        spans_path = Path(spans_output_cfg.get("file", "conversation_spans.csv"))
        spans_extend = spans_output_cfg.get("extend", False)
        write_spans(
            rows=spans_rows,
            path=spans_path,
            extend=spans_extend,
            model_name=model_name,
        )
        logger.info(f"Span records saved to {spans_path} ({len(spans_rows)} rows)")


if __name__ == "__main__":
    main()

