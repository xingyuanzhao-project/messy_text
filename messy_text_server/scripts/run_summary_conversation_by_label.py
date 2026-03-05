"""
Label-based conversation processing entry point for MessyText.

This script implements a two-stage, victim-level summarization pipeline
that separates label extraction from summary synthesis, with two
execution modes controlled by ``async.enabled`` in settings.yaml:

Hybrid mode (async.enabled: false)
    Per victim (concurrent across victims via semaphore + asyncio.gather):
        For each document SEQUENTIALLY:
            1. All labels extracted concurrently (one asyncio.gather per doc).
            2. Summary call (await) with previous_summary from prior doc,
               preserving the turn-by-turn dependency chain.
    Preserves the sequential document summary chain while maximizing
    inter-victim and intra-document-label concurrency.

Full-async mode (async.enabled: true)
    Per victim (concurrent across victims via semaphore + asyncio.gather):
        1. ALL labels × ALL documents in one asyncio.gather (max concurrency).
        2. ALL per-document summary calls concurrent (no chaining).
        3. Per-document summaries collected from orchestrator results (pure Python).
        4. ONE synthesis LLM call reconciles them into the final victim-level summary.
    Fastest possible wall-clock time; no sequential dependency anywhere.

Stage 1 — Label Extraction:
    For each document, one AsyncLabelExtractor per taxonomy label runs
    against the document text. Each extractor produces a ProcessorResult
    containing info_found, spans, and confidence_score for its label.

Stage 2 — Summary Synthesis:
    Hybrid: The per-label results are passed to
    AsyncTextLabelsSummaryProcessor turn-by-turn, with previous_summary
    chaining.  Full-async: All docs summarized concurrently via
    AsyncTextConversationOrchestrator, then per-doc summaries are collected
    and a single synthesis call (label_synthesis prompt) reconciles them into
    the final victim-level summary.

Input
-----
A report-level CSV (e.g. df_text_by_report.csv) with at least:
    - 'index'  : document identifier (do not confuse with pandas index)
    - 'victim' : victim identifier used to group documents
    - 'text'   : raw document text

Configuration
-------------
The script reads settings from config/settings.yaml and respects the
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
      enabled: true               # true=full-async, false=hybrid
      max_retries: 5
      max_concurrent_rows: 5      # max_concurrent_victims

    processing:
      temperature: 0.0
      max_tokens_summary: 1024
      max_tokens_classification: 256
      conversation:
        enabled: false             # opt-in for this script

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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as tqdm_async

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors import (  # noqa: E402
    AsyncLabelExtractor,
    AsyncTextConversationOrchestrator,
    AsyncTextLabelsSummaryProcessor,
    MessyTextConversationState,
    ProcessorResult,
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
    setup_logger,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


async def _extract_labels_for_doc_async(
    text: str,
    extractors: Dict[str, AsyncLabelExtractor],
    doc_id: Optional[Any] = None,
) -> Dict[str, ProcessorResult]:
    """
    Runs all label extractors concurrently on a single document.

    For each taxonomy label, the corresponding AsyncLabelExtractor is called
    with the document text. All extraction calls are launched simultaneously
    via asyncio.gather and results are returned in a dict keyed by label_key.

    Used by the hybrid path where documents are processed sequentially but
    each document's labels are extracted concurrently.

    Args:
        text (str): The raw document text to extract from.
        extractors (Dict[str, AsyncLabelExtractor]): Mapping from label_key
            to the pre-constructed AsyncLabelExtractor bound to that key.
        doc_id (Optional[Any]): Document identifier passed through to each
            extractor for traceability in the ProcessorResult.

    Returns:
        Dict[str, ProcessorResult]: Mapping from label_key to the extraction
        result (info_found, spans, confidence_score) for that label.
    """
    label_keys = list(extractors.keys())
    tasks = [
        extractors[key].extract_label(text=text, doc_id=doc_id)
        for key in label_keys
    ]
    results = await asyncio.gather(*tasks)
    return dict(zip(label_keys, results))


async def _extract_all_docs_labels_async(
    texts: List[str],
    doc_ids: List[Any],
    extractors: Dict[str, AsyncLabelExtractor],
    use_progress_bar: bool = True,
) -> List[Dict[str, ProcessorResult]]:
    """
    Runs all label extractors on all documents in a single asyncio.gather.

    Fires N_docs * N_labels extraction calls simultaneously, achieving
    maximum concurrency. Results are reconstructed into a list of per-document
    label_results dicts, preserving submission order.

    Used by the full-async path where no sequential dependency exists
    between documents.

    Args:
        texts (List[str]): Raw text for each document, in order.
        doc_ids (List[Any]): Document identifier for each document, in order.
        extractors (Dict[str, AsyncLabelExtractor]): Mapping from label_key
            to the pre-constructed AsyncLabelExtractor.

    Returns:
        List[Dict[str, ProcessorResult]]: One dict per document (same order
        as texts/doc_ids), each mapping label_key to its extraction result.
    """
    label_keys = list(extractors.keys())
    n_labels = len(label_keys)

    tasks = []
    for doc_id, text in zip(doc_ids, texts):
        for key in label_keys:
            tasks.append(extractors[key].extract_label(text=text, doc_id=doc_id))

    all_results = await tqdm_async.gather(
        *tasks,
        desc=f"Extracting labels ({len(texts)} docs × {n_labels} labels)",
        leave=False,
        disable=not use_progress_bar,
    )

    per_doc: List[Dict[str, ProcessorResult]] = []
    for i in range(len(texts)):
        offset = i * n_labels
        label_results = {
            label_keys[j]: all_results[offset + j]
            for j in range(n_labels)
        }
        per_doc.append(label_results)

    return per_doc


def _collect_per_doc_summaries(
    results: List[ProcessorResult],
) -> List[str]:
    """
    Extracts the per-document summary strings from a list of ProcessorResults.

    Each result corresponds to one document that was summarized independently
    by the concurrent orchestrator. This collects the 'summary' field from
    each result in order. Empty or missing summaries are included as empty
    strings so that position is preserved; the synthesizer filters them out.

    Args:
        results (List[ProcessorResult]): Per-document summary results from
            the concurrent orchestrator, in submission order.

    Returns:
        List[str]: Ordered list of summary strings, one per document.
    """
    return [result.get("summary") or "" for result in results]


def _build_processor_config(
    config: Dict,
    prompts: Dict,
) -> Dict[str, Any]:
    """
    Assembles the processor-level configuration dict from the top-level
    settings.yaml config and the loaded prompts.json.

    This is the dict passed to AsyncLabelExtractor and
    AsyncTextLabelsSummaryProcessor as their ``config`` argument. It mirrors
    the subset of settings those classes read: model name, processing
    parameters, and prompt templates.

    Args:
        config (Dict): Full configuration dictionary from settings.yaml.
        prompts (Dict): Prompt configuration loaded from config/prompts.json.

    Returns:
        Dict[str, Any]: Configuration dict with keys 'model', 'processing',
        'prompts', and 'logging'.
    """
    processing_cfg = config.get("processing", {})
    return {
        "model": {"name": config["model"]["name"]},
        "processing": {
            "temperature": processing_cfg["temperature"],
            "max_tokens_summary": processing_cfg["max_tokens_summary"],
            "max_tokens_classification": processing_cfg.get(
                "max_tokens_classification", 256
            ),
        },
        "prompts": prompts,
        "logging": config.get("logging", {}),
    }


def _build_async_extractors(
    client: AsyncOpenAI,
    processor_config: Dict[str, Any],
    taxonomy: Dict[str, Any],
    logger,
    llm_semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, AsyncLabelExtractor]:
    """
    Constructs one AsyncLabelExtractor per taxonomy label.

    Each extractor is bound to a single (label_key, label_definition) pair
    at construction time. The runner iterates taxonomy['context_definitions']
    to enumerate all labels; the extractor itself has no access to the full
    taxonomy.

    Args:
        client (AsyncOpenAI): The asynchronous OpenAI/vLLM client instance
            shared by all extractors.
        processor_config (Dict[str, Any]): Processor-level config produced by
            _build_processor_config.
        taxonomy (Dict[str, Any]): Full taxonomy dict with 'context_definitions'
            mapping label_key to its human-readable definition.
        logger: Logger instance.
        llm_semaphore (Optional[asyncio.Semaphore]): Global LLM concurrency
            limiter passed through to each extractor.

    Returns:
        Dict[str, AsyncLabelExtractor]: Mapping from label_key to the
        constructed extractor.
    """
    extractors: Dict[str, AsyncLabelExtractor] = {}
    for label_key, label_definition in taxonomy["context_definitions"].items():
        extractors[label_key] = AsyncLabelExtractor(
            client=client,
            config=processor_config,
            label_key=label_key,
            label_definition=label_definition,
            logger=logger,
            llm_semaphore=llm_semaphore,
        )
    return extractors


# ---------------------------------------------------------------------------
# Victim-level processing functions
# ---------------------------------------------------------------------------


async def _process_victim_hybrid(
    victim_id: str,
    group_df: pd.DataFrame,
    extractors: Dict[str, AsyncLabelExtractor],
    summary_processor: AsyncTextLabelsSummaryProcessor,
) -> Tuple[str, Dict[int, str], MessyTextConversationState]:
    """
    Processes all documents for a single victim using the hybrid pipeline:
    async label extraction per document, sequential document-level summary
    with await (preserving the turn-by-turn dependency chain).

    For each document in the victim's group (ordered by the 'index' column):
        1. All label extractors run concurrently on the document text via
           asyncio.gather (async within one document).
        2. The summary processor is called with await, passing
           previous_summary from the prior document's result.
        3. State is updated (turn_index, results) by this function.

    Because each summary call is awaited (not blocking), the event loop can
    serve other victims' coroutines during the wait. Cross-victim concurrency
    is handled by the caller via asyncio.Semaphore + asyncio.gather.

    Args:
        victim_id (str): Identifier for the victim (from the 'victim' column).
        group_df (pd.DataFrame): Subset of the input DataFrame for this victim,
            expected to contain at least 'index' and 'text' columns.
        extractors (Dict[str, AsyncLabelExtractor]): Mapping from label_key to
            the pre-constructed AsyncLabelExtractor for that taxonomy label.
            Shared across victims (stateless).
        summary_processor (AsyncTextLabelsSummaryProcessor): Summary processor
            called with await per document. Selects label_summary_first or
            label_summary_update based on whether previous_summary is present.

    Returns:
        Tuple[str, Dict[int, str], MessyTextConversationState]:
            - The victim_id (passthrough for downstream aggregation).
            - A mapping from pandas row index to the per-turn summary string.
            - The final MessyTextConversationState containing all turn results.
    """
    group_sorted = group_df.sort_values(by="index")
    index_list = group_sorted.index.tolist()
    texts: List[str] = [str(t) for t in group_sorted["text"]]
    doc_ids = list(group_sorted["index"])

    state = MessyTextConversationState(turn_index=0)
    per_row_summaries: List[str] = []

    for doc_id, raw_text in zip(doc_ids, texts):
        label_results = await _extract_labels_for_doc_async(
            text=raw_text,
            extractors=extractors,
            doc_id=doc_id,
        )

        result = await summary_processor.summarize_from_labels(
            text=raw_text,
            label_results=label_results,
            previous_summary=state.last_summary,
            doc_id=doc_id,
        )

        new_results = state.results.copy()
        new_results.append(result)
        state = MessyTextConversationState(
            turn_index=state.turn_index + 1,
            results=new_results,
        )

        summary = result.get("summary") or ""
        per_row_summaries.append(summary)

    index_to_summary: Dict[int, str] = dict(zip(index_list, per_row_summaries))
    return victim_id, index_to_summary, state


async def _process_victim_full_async(
    victim_id: str,
    group_df: pd.DataFrame,
    extractors: Dict[str, AsyncLabelExtractor],
    orchestrator: AsyncTextConversationOrchestrator,
    summary_processor: AsyncTextLabelsSummaryProcessor,
    use_progress_bar: bool = True,
) -> Tuple[str, Dict[int, str], MessyTextConversationState]:
    """
    Fully concurrent processing for a single victim with final synthesis.

    All work is maximally parallelized:
        1. ALL labels × ALL documents fire in one asyncio.gather call.
        2. ALL per-document summary calls fire concurrently via the
           AsyncTextConversationOrchestrator (no sequential chaining).
        3. Per-document summaries are collected from the orchestrator results
           (pure Python, no I/O).
        4. ONE synthesis LLM call produces the final victim-level summary
           by reconciling all per-document summaries.

    The synthesized summary is written to every row for this victim (since
    there is no meaningful per-document intermediate in the concurrent model).

    Args:
        victim_id (str): Identifier for the victim (from the 'victim' column).
        group_df (pd.DataFrame): Subset of the input DataFrame for this victim,
            expected to contain at least 'index' and 'text' columns.
        extractors (Dict[str, AsyncLabelExtractor]): Mapping from label_key to
            the pre-constructed AsyncLabelExtractor for that taxonomy label.
        orchestrator (AsyncTextConversationOrchestrator): Concurrent orchestrator
            that fires all document summary calls at once via asyncio.gather.
        summary_processor (AsyncTextLabelsSummaryProcessor): Summary processor
            also used for the final synthesis call (synthesize_from_summaries).

    Returns:
        Tuple[str, Dict[int, str], MessyTextConversationState]:
            - The victim_id (passthrough for downstream aggregation).
            - A mapping from pandas row index to the synthesized summary
              (same value for all rows of this victim).
            - The final MessyTextConversationState containing per-document
              results plus the synthesis result appended at the end.
    """
    group_sorted = group_df.sort_values(by="index")
    index_list = group_sorted.index.tolist()
    texts: List[str] = [str(t) for t in group_sorted["text"]]
    doc_ids = list(group_sorted["index"])

    per_doc_label_results = await _extract_all_docs_labels_async(
        texts=texts,
        doc_ids=doc_ids,
        extractors=extractors,
        use_progress_bar=use_progress_bar,
    )

    documents: List[Tuple[str, Dict[str, ProcessorResult], Any]] = [
        (text, label_results, doc_id)
        for text, label_results, doc_id in zip(texts, per_doc_label_results, doc_ids)
    ]

    _per_doc_summaries, state = await orchestrator.run_conversation(
        documents=documents,
        use_progress_bar=use_progress_bar,
    )

    per_doc_summaries = _collect_per_doc_summaries(state.results)

    synthesis_result = await summary_processor.synthesize_from_summaries(
        per_doc_summaries=per_doc_summaries,
        doc_id=victim_id,
    )

    new_results = state.results.copy()
    new_results.append(synthesis_result)
    state = MessyTextConversationState(
        turn_index=state.turn_index + 1,
        results=new_results,
    )

    final_summary = synthesis_result.get("summary") or ""
    index_to_summary: Dict[int, str] = {
        row_idx: final_summary for row_idx in index_list
    }

    return victim_id, index_to_summary, state


# ---------------------------------------------------------------------------
# DataFrame-level processing functions
# ---------------------------------------------------------------------------


async def _process_dataframe_hybrid(
    df: pd.DataFrame,
    async_client: AsyncOpenAI,
    config: Dict,
    prompts: Dict,
    logger,
) -> Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
    """
    Hybrid processing: async labels + async victims, sequential documents.

    Per victim (concurrent across victims via semaphore + asyncio.gather):
        - For each document sequentially:
            1. Extract all labels concurrently (async within one document).
            2. Await summary call, passing previous_summary from prior doc.
        - Preserves the turn-by-turn dependency chain (label_summary_first
          on first doc, label_summary_update on subsequent docs).

    This mode maximizes inter-victim concurrency while preserving the
    sequential document-level summary chain that allows each document's
    summary to build on the previous one.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'index', 'victim',
            and 'text' columns (at minimum).
        async_client (AsyncOpenAI): Asynchronous OpenAI/vLLM client used for
            both label extraction and summary calls.
        config (Dict): Full configuration dictionary from settings.yaml.
        prompts (Dict): Prompt configuration loaded from config/prompts.json.
        logger: Logger instance.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
            - Updated DataFrame with a 'summary_all_context' column containing
              the per-turn conversation summaries.
            - List of (victim_id, final_state) tuples for record serialization.
    """
    processing_cfg = config.get("processing", {})
    async_cfg = config.get("async", {}) or {}

    max_concurrent_victims = async_cfg.get("max_concurrent_rows", 5)
    max_concurrent_llm = async_cfg.get("max_concurrent_llm_calls")
    use_progress_bar = config.get("display", {}).get("use_progress_bar", True)
    summary_row_limit = processing_cfg.get("summary_row_limit")

    llm_semaphore: Optional[asyncio.Semaphore] = (
        asyncio.Semaphore(max_concurrent_llm) if max_concurrent_llm else None
    )

    taxonomy_path = config["paths"]["taxonomy"]
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    processor_config = _build_processor_config(config, prompts)

    extractors = _build_async_extractors(
        client=async_client,
        processor_config=processor_config,
        taxonomy=taxonomy,
        logger=logger,
        llm_semaphore=llm_semaphore,
    )

    summary_processor = AsyncTextLabelsSummaryProcessor(
        client=async_client,
        config=processor_config,
        taxonomy=taxonomy,
        logger=logger,
        llm_semaphore=llm_semaphore,
    )

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        df_processed["summary_all_context"] = ""

    victim_groups = list(df_processed.groupby("victim"))
    if summary_row_limit is not None:
        victim_groups = victim_groups[:summary_row_limit]

    semaphore = asyncio.Semaphore(max_concurrent_victims)
    tasks: List[asyncio.Future] = []

    for victim_id, group_df in victim_groups:

        async def _bounded_task(v_id=victim_id, g_df=group_df):
            async with semaphore:
                return await _process_victim_hybrid(
                    v_id,
                    g_df,
                    extractors,
                    summary_processor,
                )

        tasks.append(_bounded_task())

    results: List[Tuple[str, Dict[int, str], MessyTextConversationState]] = []
    for coro in tqdm_async.as_completed(
        tasks,
        total=len(tasks),
        desc="Processing victims (hybrid)",
        disable=not use_progress_bar,
    ):
        result = await coro
        results.append(result)

    for _victim_id, index_to_summary, _state in results:
        for row_idx, summary in index_to_summary.items():
            df_processed.at[row_idx, "summary_all_context"] = summary

    victim_states: List[Tuple[str, MessyTextConversationState]] = [
        (victim_id, state) for victim_id, _summary_map, state in results
    ]

    return df_processed, victim_states


async def _process_dataframe_full_async(
    df: pd.DataFrame,
    async_client: AsyncOpenAI,
    config: Dict,
    prompts: Dict,
    logger,
) -> Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
    """
    Full-async processing: everything concurrent, reconcile with synthesis.

    Per victim (concurrent across victims via semaphore + asyncio.gather):
        1. ALL labels × ALL documents in one asyncio.gather (max concurrency).
        2. ALL per-document summary calls concurrent (no chaining).
        3. Per-document summaries collected from orchestrator results (pure Python).
        4. ONE synthesis LLM call reconciles them into the final victim-level summary.

    The synthesized summary is written to every row for the victim. Per-document
    results are preserved in the state for record serialization.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'index', 'victim',
            and 'text' columns (at minimum).
        async_client (AsyncOpenAI): Asynchronous OpenAI/vLLM client used for
            label extraction, per-document summary, and synthesis calls.
        config (Dict): Full configuration dictionary from settings.yaml.
        prompts (Dict): Prompt configuration loaded from config/prompts.json.
        logger: Logger instance.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[str, MessyTextConversationState]]]:
            - Updated DataFrame with a 'summary_all_context' column containing
              the synthesized victim-level summary for each row.
            - List of (victim_id, final_state) tuples for record serialization.
    """
    processing_cfg = config.get("processing", {})
    async_cfg = config.get("async", {}) or {}

    max_concurrent_victims = async_cfg.get("max_concurrent_rows", 5)
    max_concurrent_llm = async_cfg.get("max_concurrent_llm_calls")
    use_progress_bar = config.get("display", {}).get("use_progress_bar", True)
    summary_row_limit = processing_cfg.get("summary_row_limit")

    llm_semaphore: Optional[asyncio.Semaphore] = (
        asyncio.Semaphore(max_concurrent_llm) if max_concurrent_llm else None
    )

    taxonomy_path = config["paths"]["taxonomy"]
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    processor_config = _build_processor_config(config, prompts)

    extractors = _build_async_extractors(
        client=async_client,
        processor_config=processor_config,
        taxonomy=taxonomy,
        logger=logger,
        llm_semaphore=llm_semaphore,
    )

    summary_processor = AsyncTextLabelsSummaryProcessor(
        client=async_client,
        config=processor_config,
        taxonomy=taxonomy,
        logger=logger,
        llm_semaphore=llm_semaphore,
    )
    orchestrator = AsyncTextConversationOrchestrator(summary_processor)

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        df_processed["summary_all_context"] = ""

    victim_groups = list(df_processed.groupby("victim"))
    if summary_row_limit is not None:
        victim_groups = victim_groups[:summary_row_limit]

    semaphore = asyncio.Semaphore(max_concurrent_victims)
    tasks: List[asyncio.Future] = []

    for victim_id, group_df in victim_groups:

        async def _bounded_task(v_id=victim_id, g_df=group_df):
            async with semaphore:
                return await _process_victim_full_async(
                    v_id,
                    g_df,
                    extractors,
                    orchestrator,
                    summary_processor,
                    use_progress_bar=use_progress_bar,
                )

        tasks.append(_bounded_task())

    results: List[Tuple[str, Dict[int, str], MessyTextConversationState]] = []
    for coro in tqdm_async.as_completed(
        tasks,
        total=len(tasks),
        desc="Processing victims (full-async)",
        disable=not use_progress_bar,
    ):
        result = await coro
        results.append(result)

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
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    
    logger = setup_logger(log_file=settings["logging"]["file"])
    
    # Load prompt configuration
    prompts_path = settings["paths"].get("prompts", "config/prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    processing_cfg = settings.get("processing", {})
    conversation_cfg = processing_cfg.get("conversation", {}) or {}
    conversation_enabled = conversation_cfg.get("enabled", False)

    if not conversation_enabled:
        logger.info(
            "Conversation processing is disabled in config (processing.conversation.enabled=false). "
            "Exiting without changes."
        )
        return

    # Step 2: Load data (conversation = summary-style processing)
    summary_paths = settings["paths"]["summary"]
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
        base_url=settings["model"]["api_base"],
        api_key=settings["model"]["api_key"],
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
        settings["model"]["name"],
        logger,
    )

    if not vllm_ok:
        logger.error(f"Pre-flight FAILED. Models found: {available_models}")
        sys.exit(1)

    logger.info("Pre-flight checks PASSED")
    logger.info("=" * 50)

    # Step 4: Select processing mode
    #   async.enabled=true  → full-async (all concurrent + synthesis)
    #   async.enabled=false → hybrid (async labels + victims, sequential docs)
    async_enabled = settings.get("async", {}).get("enabled", True)
    max_retries = settings.get("async", {}).get("max_retries", 2)

    async_client = AsyncOpenAI(
        base_url=settings["model"]["api_base"],
        api_key=settings["model"]["api_key"],
        max_retries=max_retries,
    )

    _llm_cap = settings.get("async", {}).get("max_concurrent_llm_calls")
    logger.info(f"LLM concurrency cap: {_llm_cap or 'unlimited'}")

    if async_enabled:
        logger.info("Using FULL-ASYNC processing (all concurrent + synthesis).")
        processed_df, victim_states = asyncio.run(
            _process_dataframe_full_async(
                df=df_text,
                async_client=async_client,
                config=settings,
                prompts=prompts,
                logger=logger,
            )
        )
    else:
        logger.info("Using HYBRID processing (async labels + victims, sequential docs).")
        processed_df, victim_states = asyncio.run(
            _process_dataframe_hybrid(
                df=df_text,
                async_client=async_client,
                config=settings,
                prompts=prompts,
                logger=logger,
            )
        )

    # Step 5: Save output
    model_name = settings["model"]["name"]
    processed_df["model"] = model_name
    if "summary_all_context" in processed_df.columns:
        processed_df["summary_all_context"].replace(
            ["No information", "No relevant information found"],
            "",
            inplace=True,
        )

    summary_output_paths = settings["paths"]["summary"]["output"]
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
    records_cfg = settings["paths"].get("records", {})
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

