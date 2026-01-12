"""
Standalone classification entry point for MessyText (non-conversational).

This script reuses the existing MessyTextProcessor classification logic and
configuration from run_processing.py, but assumes that per-row summaries are
already present in the DataFrame (column: 'summary_all_context').

Intended usage:

- run_summary.py         : compute summary_all_context from raw text
- run_classification.py  : take those summaries and produce *_classification columns

No new prompt or parsing logic is introduced here; all details are delegated
to MessyTextProcessor.classify_summary and the existing taxonomy + prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import time
import gc

import pandas as pd
import yaml
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as tqdm_async
from tqdm.auto import tqdm

from src.processors import AsyncMessyTextProcessor, MessyTextProcessor
from src.utils import check_gpu_info, check_vllm_server, log_memory_usage, setup_logger


def _process_dataframe_classification_sync(
    df: pd.DataFrame,
    vllm_client: OpenAI,
    model_name: str,
    prompts_config: dict,
    log_file: str = "processing.log",
    start_index: int = 0,
    summary_row_limit: int | None = None,
    classification_key_limit: int | None = None,
    log_resources: bool = False,
    log_response: bool = False,
    log_progress: bool = False,
    use_progress_bar: bool = True,
    max_tokens_classification: int = 256,
) -> pd.DataFrame:
    """
    Synchronous classification-only pipeline.

    Mirrors the classification part of process_dataframe_summary_and_classification
    from run_processing.py, but:
      - Assumes 'summary_all_context' is already populated.
      - Does not call summarize_text.
    """
    logger = setup_logger(log_file=log_file)

    # Load taxonomy (same structure as in run_processing.py)
    with open("config/taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    code_to_desc_map = taxonomy["context_definitions"]
    label_values_map = taxonomy["label_options"]

    config = {
        "model": {"name": model_name},
        "processing": {
            "temperature": 0.0,
            "max_tokens_summary": 0,  # unused here
            "max_tokens_classification": max_tokens_classification,
        },
        "prompts": prompts_config,
    }

    taxonomy_for_processor = {
        "context_definitions": code_to_desc_map,
        "label_options": label_values_map,
    }

    processor = MessyTextProcessor(vllm_client, config, taxonomy_for_processor, logger)

    all_keys = list(code_to_desc_map.keys())
    keys_to_classify = all_keys[:classification_key_limit] if classification_key_limit else all_keys

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        raise ValueError(
            "Input DataFrame must contain 'summary_all_context' for classification-only processing."
        )

    # Ensure all classification columns exist
    for key in all_keys:
        col_name = f"{key}_classification"
        if col_name not in df_processed.columns:
            df_processed[col_name] = ""

    df_to_process = df_processed.iloc[start_index:]
    total_rows = len(df_to_process)
    if summary_row_limit is not None and summary_row_limit < total_rows:
        total_rows = summary_row_limit

    results_list: list[dict] = []
    row_counter = 0
    start_time = time.time()
    logged_milestones: set[int] = set()

    with tqdm(
        total=total_rows,
        desc="Classifying",
        position=0,
        leave=True,
        disable=not use_progress_bar,
    ) as pbar:
        for row in df_to_process.itertuples():
            if summary_row_limit is not None and row_counter >= summary_row_limit:
                break

            pbar.set_description(f"Classifying (Index: {row.Index})")

            if log_progress:
                logger.info(f"Classifying row {row.Index}")

            current_row_results: dict = {"index": row.Index}

            # Use existing summary as input to classification.
            summary = getattr(row, "summary_all_context", "")

            if summary and summary != "No relevant information found":
                with tqdm(
                    keys_to_classify,
                    total=len(keys_to_classify),
                    desc="Classifying keys",
                    leave=False,
                    position=1,
                    disable=not use_progress_bar,
                ) as pbar_inner:
                    for key in pbar_inner:
                        classification = processor.classify_summary(summary, key)
                        current_row_results[f"{key}_classification"] = classification

                        if log_response:
                            logger.info(f"  {key}: {classification}")

            # Ensure all keys are present even if classification_key_limit on keys is used.
            for key in all_keys:
                col_name = f"{key}_classification"
                if col_name not in current_row_results:
                    current_row_results[col_name] = df_processed.at[row.Index, col_name]

            results_list.append(current_row_results)

            if log_resources:
                log_memory_usage(logger, f"After classification row {row.Index}")

            # No large temporary allocations here, but keep GC behavior consistent.
            gc.collect()

            row_counter += 1
            pbar.update(1)

            if not use_progress_bar and total_rows > 0:
                pct = int((row_counter / total_rows) * 100)
                milestone = (pct // 10) * 10
                if milestone > 0 and milestone not in logged_milestones:
                    elapsed = time.time() - start_time
                    rate = row_counter / elapsed if elapsed > 0 else 0
                    eta = (total_rows - row_counter) / rate if rate > 0 else 0
                    logger.info(
                        f"Index {row.Index} | {milestone}% ({row_counter}/{total_rows}) | "
                        f"Elapsed: {elapsed:.0f}s | {rate:.2f} rows/sec | ETA: {eta:.0f}s"
                    )
                    logged_milestones.add(milestone)

            if row_counter % 5 == 0:
                time.sleep(2)

    if results_list:
        results_df = pd.DataFrame(results_list).set_index("index")
        df_processed.update(results_df)

    logger.info(f"Classification complete. Processed {row_counter} rows.")
    return df_processed


async def _classify_row_async(
    row,
    processor: AsyncMessyTextProcessor,
    keys_to_classify,
    all_keys,
    semaphore,
):
    """
    Asynchronous per-row classification helper.

    Mirrors the classification part of process_row_async in run_processing.py,
    but uses an existing summary_all_context instead of recomputing summaries.
    """
    async with semaphore:
        current_row_results: dict = {"index": row.Index}

        summary = getattr(row, "summary_all_context", "")

        if summary and summary != "No relevant information found":
            tasks = []
            for key in keys_to_classify:
                tasks.append(processor.classify_summary(summary, key))

            results = await asyncio.gather(*tasks)

            for key, classification in zip(keys_to_classify, results):
                current_row_results[f"{key}_classification"] = classification

        for key in all_keys:
            col_name = f"{key}_classification"
            if col_name not in current_row_results:
                current_row_results[col_name] = ""

        return current_row_results


async def _process_dataframe_classification_async(
    df: pd.DataFrame,
    vllm_client: AsyncOpenAI,
    model_name: str,
    prompts_config: dict,
    log_file: str = "processing.log",
    start_index: int = 0,
    summary_row_limit: int | None = None,
    classification_key_limit: int | None = None,
    log_resources: bool = False,
    use_progress_bar: bool = True,
    max_concurrent_rows: int = 50,
    max_tokens_classification: int = 256,
) -> pd.DataFrame:
    """
    Asynchronous classification-only pipeline.

    Reuses AsyncMessyTextProcessor.classify_summary with the same taxonomy and
    prompts as run_processing.py, but assumes 'summary_all_context' is present.
    """
    logger = setup_logger(log_file=log_file)

    with open("config/taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    code_to_desc_map = taxonomy["context_definitions"]
    label_values_map = taxonomy["label_options"]

    config = {
        "model": {"name": model_name},
        "processing": {
            "temperature": 0.0,
            "max_tokens_summary": 0,
            "max_tokens_classification": max_tokens_classification,
        },
        "prompts": prompts_config,
    }

    taxonomy_for_processor = {
        "context_definitions": code_to_desc_map,
        "label_options": label_values_map,
    }

    processor = AsyncMessyTextProcessor(vllm_client, config, taxonomy_for_processor, logger)

    all_keys = list(code_to_desc_map.keys())
    keys_to_classify = all_keys[:classification_key_limit] if classification_key_limit else all_keys

    df_processed = df.copy()
    if "summary_all_context" not in df_processed.columns:
        raise ValueError(
            "Input DataFrame must contain 'summary_all_context' for classification-only processing."
        )

    # Initialize classification columns
    for key in all_keys:
        col_name = f"{key}_classification"
        if col_name not in df_processed.columns:
            df_processed[col_name] = ""

    df_to_process = df_processed.iloc[start_index:]
    total_rows = len(df_to_process)
    if summary_row_limit is not None and summary_row_limit < total_rows:
        df_to_process = df_to_process.iloc[:summary_row_limit]
        total_rows = summary_row_limit

    semaphore = asyncio.Semaphore(max_concurrent_rows)

    logger.info(
        f"Starting async classification with {max_concurrent_rows} concurrent rows."
    )

    tasks = []
    for row in df_to_process.itertuples():
        tasks.append(
            _classify_row_async(
                row=row,
                processor=processor,
                keys_to_classify=keys_to_classify,
                all_keys=all_keys,
                semaphore=semaphore,
            )
        )

    results_list: list[dict] = []
    for coro in tqdm_async.as_completed(
        tasks,
        total=len(tasks),
        desc="Classifying Rows",
        disable=not use_progress_bar,
    ):
        result = await coro
        results_list.append(result)
        if log_resources and len(results_list) % 100 == 0:
            log_memory_usage(logger, f"Classified {len(results_list)} rows")

    if results_list:
        results_df = pd.DataFrame(results_list).set_index("index")
        df_processed.update(results_df)

    logger.info(f"Classification complete. Processed {len(results_list)} rows.")
    return df_processed


def main() -> None:
    """
    Main entry point for non-conversational classification-only processing.

    Behavior mirrors run_processing.py:
        - Reads config/settings.yaml
        - Runs GPU + vLLM pre-flight checks
        - Chooses async vs sync based on async.enabled
        - Writes classification columns based on 'summary_all_context'
    """
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    logger = setup_logger(log_file=settings["logging"]["file"])

    prompts_path = settings["paths"].get("prompts", "config/prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_config = json.load(f)

    # Load data from classification input path
    classification_paths = settings["paths"]["classification"]
    df_text = pd.read_csv(classification_paths["input"], encoding="utf-8")

    # Pre-flight checks using a synchronous client
    sync_client = OpenAI(
        base_url=settings["model"]["api_base"],
        api_key=settings["model"]["api_key"],
    )

    logger.info("=" * 50)
    logger.info("PRE-FLIGHT CHECKS (Classification-only)")
    logger.info("=" * 50)

    gpu_info = check_gpu_info(logger)
    if gpu_info is None:
        logger.error("No GPU available. Exiting.")
        sys.exit(1)

    vllm_ok, available_models, _ = check_vllm_server(
        sync_client,
        settings["model"]["name"],
        logger,
    )

    if not vllm_ok:
        logger.error(f"Pre-flight FAILED. Models found: {available_models}")
        sys.exit(1)

    logger.info("Pre-flight checks PASSED")
    logger.info("=" * 50)

    async_processing = settings["async"].get("enabled", True)

    if async_processing:
        logger.info("Using ASYNC classification mode.")
        max_retries = settings["async"].get("max_retries", 2)
        max_concurrent_rows = settings["async"].get("max_concurrent_rows", 50)

        async_client = AsyncOpenAI(
            base_url=settings["model"]["api_base"],
            api_key=settings["model"]["api_key"],
            max_retries=max_retries,
        )

        processed_df = asyncio.run(
            _process_dataframe_classification_async(
                df=df_text,
                vllm_client=async_client,
                model_name=settings["model"]["name"],
                prompts_config=prompts_config,
                log_file=settings["logging"]["file"],
                start_index=settings["processing"]["start_index"],
                summary_row_limit=settings["processing"].get("summary_row_limit"),
                classification_key_limit=settings["processing"].get("classification_key_limit"),
                log_resources=settings["logging"]["log_resources"],
                use_progress_bar=settings["display"]["use_progress_bar"],
                max_concurrent_rows=max_concurrent_rows,
                max_tokens_classification=settings["processing"].get(
                    "max_tokens_classification", 256
                ),
            )
        )
    else:
        logger.info("Using SYNC classification mode.")
        processed_df = _process_dataframe_classification_sync(
            df=df_text,
            vllm_client=sync_client,
            model_name=settings["model"]["name"],
            prompts_config=prompts_config,
            log_file=settings["logging"]["file"],
            start_index=settings["processing"]["start_index"],
            summary_row_limit=settings["processing"].get("summary_row_limit"),
            classification_key_limit=settings["processing"].get("classification_key_limit"),
            log_resources=settings["logging"]["log_resources"],
            log_response=settings["logging"]["log_response"],
            log_progress=settings["logging"]["log_progress"],
            use_progress_bar=settings["display"]["use_progress_bar"],
            max_tokens_classification=settings["processing"].get(
                "max_tokens_classification", 256
            ),
        )

    model_name = settings["model"]["name"]
    processed_df["model"] = model_name

    # Normalize "no information" markers to empty strings like run_processing.py
    processed_df.replace(
        ["No information", "No relevant information found"],
        "",
        inplace=True,
    )

    classification_output_paths = settings["paths"]["classification"]["output"]
    output_path = classification_output_paths["file"]
    extend_mode = classification_output_paths.get("extend", False)

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


if __name__ == "__main__":
    main()

