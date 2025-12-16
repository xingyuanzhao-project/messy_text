"""
Evaluation script for summarization quality.
Reads output CSV from config and evaluates summary quality using registered metrics.
Outputs results with all original columns plus evaluation scores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time
import pandas as pd
import yaml
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm as tqdm_async
from openai import OpenAI, AsyncOpenAI

from sklearn.metrics import f1_score, cohen_kappa_score

from src.metrics import GEvalEvaluator, AsyncGEvalEvaluator, SummaCEvaluator, DefaultMetricsEvaluator
from src.utils import setup_logger, check_vllm_server


# =============================================================================
# Parent Helper Function: Shared Evaluation Logic
# =============================================================================

async def _evaluate_single_row_async(
    idx,
    source,
    summary,
    eval_method,
    semaphore,
    log_progress,
    logger
):
    """
    Async helper to evaluate a single row with semaphore concurrency control.
    
    Args:
        idx: Row index for logging.
        source (str): Source text.
        summary (str): Summary text.
        eval_method: Async evaluation method to call.
        semaphore (asyncio.Semaphore): Concurrency limiter.
        log_progress (bool): Enable verbose logging.
        logger: Logger instance.
    
    Returns:
        Tuple[int, float]: (row_index, score)
    """
    async with semaphore:
        if log_progress:
            logger.info(f"--- Evaluating row {idx} ---")
            logger.info(f"Source length: {len(source)} chars")
            logger.info(f"Summary length: {len(summary)} chars")
        
        score = await eval_method(source, summary)
        
        if log_progress:
            logger.info(f"Row {idx} score: {score:.3f}")
        
        return (idx, score)


async def _evaluate_metric_async(
    df_eval,
    evaluator,
    eval_method_name,
    pbar_desc,
    use_progress_bar,
    log_progress,
    logger,
    max_concurrent=10
):
    """
    Async evaluation runner. Creates all tasks upfront and runs them concurrently.
    Follows main.py pattern: tasks created first, then run via tqdm_async.as_completed().
    
    Args:
        df_eval (pd.DataFrame): DataFrame to evaluate.
        evaluator: Async evaluator instance.
        eval_method_name (str): Name of async method to call.
        pbar_desc (str): Progress bar description.
        use_progress_bar (bool): Enable progress bar.
        log_progress (bool): Enable verbose logging.
        logger: Logger instance.
        max_concurrent (int): Maximum concurrent tasks (semaphore limit).
    
    Returns:
        Dict[int, float]: Mapping of row index to score.
    """
    eval_method = getattr(evaluator, eval_method_name)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Step 1: Create all tasks upfront (like main.py lines 345-348)
    tasks = []
    for idx, row in df_eval.iterrows():
        source = str(row['text'])
        summary = str(row['summary_all_context'])
        task = _evaluate_single_row_async(
            idx, source, summary, eval_method, semaphore, log_progress, logger
        )
        tasks.append(task)
    
    # Step 2: Run tasks concurrently with progress (like main.py lines 351-354)
    results = {}
    for coro in tqdm_async.as_completed(tasks, total=len(tasks), desc=pbar_desc, disable=not use_progress_bar):
        idx, score = await coro
        results[idx] = score
    
    return results


def _evaluate_metric(
    df,
    config,
    log_file,
    rows_eval_break,
    log_progress,
    use_progress_bar,
    async_processing,
    metric_title,
    pbar_desc,
    evaluator_factory,
    eval_method_name,
    output_column,
    logger,
    max_concurrent=10
):
    """
    Parent helper function containing shared evaluation logic.
    Called by child functions that specify the metric-specific parameters.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'text' and 'summary_all_context' columns.
        config (Dict[str, Any]): Runtime settings.
        log_file (str): Path to the log file.
        rows_eval_break (Optional[int]): Stop after N rows (for testing).
        log_progress (bool): Enable verbose per-row logging.
        use_progress_bar (bool): Use tqdm progress bar.
        async_processing (bool): Use async evaluator.
        metric_title (str): Title for log output (e.g., "G-EVAL SUMMARIZATION").
        pbar_desc (str): Progress bar description (e.g., "G-Eval Sum").
        evaluator_factory (Callable): Factory function that returns evaluator instance.
        eval_method_name (str): Name of evaluation method to call (e.g., "evaluate_summarization").
        output_column (str): Column name for output scores (e.g., "geval_summarization_score").
        logger: Logger instance.
        max_concurrent (int): Maximum concurrent async tasks. Defaults to 10.

    Returns:
        pd.DataFrame: The evaluated DataFrame with new score column.
    """
    logger.info("=" * 50)
    logger.info(f"{metric_title} EVALUATION")
    logger.info("=" * 50)
    
    # Filter rows with valid summaries
    df_valid = df[
        (df['text'].notna()) & 
        (df['text'] != '') &
        (df['summary_all_context'].notna()) & 
        (df['summary_all_context'] != '')
    ]
    
    # Apply row limit if set (None = all rows)
    if rows_eval_break:
        df_eval = df_valid.head(rows_eval_break).copy()
    else:
        df_eval = df_valid.copy()
    
    total_rows = len(df_eval)
    logger.info(f"Evaluating {total_rows} rows with valid summaries")
    
    start_time = time.time()
    
    if async_processing:
        # Async mode: batch all tasks, run concurrently (like main.py)
        logger.info(f"Mode: async (max_concurrent={max_concurrent})")
        
        evaluator = evaluator_factory()
        
        # Run async evaluation loop
        results = asyncio.run(_evaluate_metric_async(
            df_eval=df_eval,
            evaluator=evaluator,
            eval_method_name=eval_method_name,
            pbar_desc=pbar_desc,
            use_progress_bar=use_progress_bar,
            log_progress=log_progress,
            logger=logger,
            max_concurrent=max_concurrent
        ))
        
        # Map scores back to DataFrame order
        scores = [results.get(idx, 0.0) for idx in df_eval.index]
    else:
        # Sync mode: sequential processing with progress bar
        logger.info("Mode: sync")
        
        evaluator = evaluator_factory()
        eval_method = getattr(evaluator, eval_method_name)
        
        scores = []
        row_counter = 0
        logged_milestones = set()
        
        with tqdm(total=total_rows, desc=pbar_desc, disable=not use_progress_bar) as pbar:
            for idx, row in df_eval.iterrows():
                pbar.set_description(f"{pbar_desc} (Index: {idx})")
                
                source = str(row['text'])
                summary = str(row['summary_all_context'])
                
                if log_progress:
                    logger.info(f"--- Evaluating row {idx} ---")
                    logger.info(f"Source length: {len(source)} chars")
                    logger.info(f"Summary length: {len(summary)} chars")
                
                score = eval_method(source, summary)
                scores.append(score)
                
                if log_progress:
                    logger.info(f"Row {idx} score: {score:.3f}")
                
                row_counter += 1
                pbar.update(1)
                
                # Milestone logging when progress bar disabled (server mode)
                if not use_progress_bar and total_rows > 0:
                    pct = int((row_counter / total_rows) * 100)
                    milestone = (pct // 10) * 10
                    if milestone > 0 and milestone not in logged_milestones:
                        elapsed = time.time() - start_time
                        rate = row_counter / elapsed if elapsed > 0 else 0
                        eta = (total_rows - row_counter) / rate if rate > 0 else 0
                        logger.info(f"Index {idx} | {milestone}% ({row_counter}/{total_rows}) | "
                                   f"Elapsed: {elapsed:.0f}s | {rate:.2f} rows/sec | ETA: {eta:.0f}s")
                        logged_milestones.add(milestone)
    
    # Add score column to dataframe
    df_eval[output_column] = scores
    
    # Summary statistics
    elapsed_total = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"{metric_title} RESULTS")
    logger.info("=" * 50)
    
    if scores:
        logger.info(f"Rows evaluated: {len(scores)}")
        logger.info(f"Total time: {elapsed_total:.1f}s")
        
        # Log per-model stats
        for model_name in df_eval['model'].unique():
            model_scores = df_eval[df_eval['model'] == model_name][output_column].tolist()
            if model_scores:
                logger.info(f"[{model_name}] Avg={sum(model_scores)/len(model_scores):.3f} | "
                           f"Min={min(model_scores):.3f} | Max={max(model_scores):.3f} | N={len(model_scores)}")
    else:
        logger.warning("No valid rows found for evaluation")
    
    return df_eval


# =============================================================================
# Child Function 1: G-Eval Summarization
# =============================================================================

def evaluate_geval_summarization(
    df,
    config,
    log_file="processing.log",
    rows_eval_break=None,
    log_progress=False,
    use_progress_bar=True,
    async_processing=False,
    max_concurrent=10
):
    """
    Evaluates summarization quality using G-Eval.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'geval_summarization_score' column.
    """
    logger = setup_logger(log_file=log_file)
    model_config = config['model']
    
    # Fallback: openai/ model or vLLM client check, else skip
    if not model_config.get('name', '').startswith('openai/'):
        client = OpenAI(base_url=model_config['api_base'], api_key=model_config['api_key'])
        vllm_ok, _, _ = check_vllm_server(client, model_config['name'], logger)
        if not vllm_ok:
            return df
    
    # Factory to create evaluator
    def evaluator_factory():
        if async_processing:
            client = AsyncOpenAI(
                base_url=model_config['api_base'],
                api_key=model_config['api_key']
            )
            return AsyncGEvalEvaluator(client, config, logger)
        else:
            client = OpenAI(
                base_url=model_config['api_base'],
                api_key=model_config['api_key']
            )
            return GEvalEvaluator(client, config, logger)
    
    return _evaluate_metric(
        df=df,
        config=config,
        log_file=log_file,
        rows_eval_break=rows_eval_break,
        log_progress=log_progress,
        use_progress_bar=use_progress_bar,
        async_processing=async_processing,
        metric_title="G-EVAL SUMMARIZATION",
        pbar_desc="G-Eval Sum",
        evaluator_factory=evaluator_factory,
        eval_method_name="evaluate_summarization",
        output_column="geval_summarization_score",
        logger=logger,
        max_concurrent=max_concurrent
    )


# =============================================================================
# Child Function 2: G-Eval Hallucination
# =============================================================================

def evaluate_geval_hallucination(
    df,
    config,
    log_file="processing.log",
    rows_eval_break=None,
    log_progress=False,
    use_progress_bar=True,
    async_processing=False,
    max_concurrent=10
):
    """
    Evaluates hallucination using G-Eval.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'geval_hallucination_score' column.
    """
    logger = setup_logger(log_file=log_file)
    model_config = config['model']
    
    # Fallback: openai/ model or vLLM client check, else skip
    if not model_config.get('name', '').startswith('openai/'):
        client = OpenAI(base_url=model_config['api_base'], api_key=model_config['api_key'])
        vllm_ok, _, _ = check_vllm_server(client, model_config['name'], logger)
        if not vllm_ok:
            return df
    
    # Factory to create evaluator
    def evaluator_factory():
        if async_processing:
            client = AsyncOpenAI(
                base_url=model_config['api_base'],
                api_key=model_config['api_key']
            )
            return AsyncGEvalEvaluator(client, config, logger)
        else:
            client = OpenAI(
                base_url=model_config['api_base'],
                api_key=model_config['api_key']
            )
            return GEvalEvaluator(client, config, logger)
    
    return _evaluate_metric(
        df=df,
        config=config,
        log_file=log_file,
        rows_eval_break=rows_eval_break,
        log_progress=log_progress,
        use_progress_bar=use_progress_bar,
        async_processing=async_processing,
        metric_title="G-EVAL HALLUCINATION",
        pbar_desc="G-Eval Hal",
        evaluator_factory=evaluator_factory,
        eval_method_name="evaluate_hallucination",
        output_column="geval_hallucination_score",
        logger=logger,
        max_concurrent=max_concurrent
    )


# =============================================================================
# Child Function 3: SummaC Zero-Shot
# =============================================================================

def evaluate_summac_zs(
    df,
    config,
    log_file="processing.log",
    rows_eval_break=None,
    log_progress=False,
    use_progress_bar=True,
    async_processing=False,  # Unused, kept for interface consistency
    max_concurrent=10  # Unused, kept for interface consistency
):
    """
    Evaluates factual consistency using SummaC Zero-Shot.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'summac_zs_score' column.
    """
    logger = setup_logger(log_file=log_file)
    
    # Factory to create evaluator
    def evaluator_factory():
        return SummaCEvaluator(config, logger)
    
    return _evaluate_metric(
        df=df,
        config=config,
        log_file=log_file,
        rows_eval_break=rows_eval_break,
        log_progress=log_progress,
        use_progress_bar=use_progress_bar,
        async_processing=False,  # SummaC is sync only
        metric_title="SUMMAC ZERO-SHOT",
        pbar_desc="SummaC ZS",
        evaluator_factory=evaluator_factory,
        eval_method_name="evaluate_zs",
        output_column="summac_zs_score",
        logger=logger,
        max_concurrent=max_concurrent
    )


# =============================================================================
# Child Function 4: SummaC Convolutional
# =============================================================================

def evaluate_summac_conv(
    df,
    config,
    log_file="processing.log",
    rows_eval_break=None,
    log_progress=False,
    use_progress_bar=True,
    async_processing=False,  # Unused, kept for interface consistency
    max_concurrent=10  # Unused, kept for interface consistency
):
    """
    Evaluates factual consistency using SummaC Convolutional.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'summac_conv_score' column.
    """
    logger = setup_logger(log_file=log_file)
    
    # Factory to create evaluator
    def evaluator_factory():
        return SummaCEvaluator(config, logger)
    
    return _evaluate_metric(
        df=df,
        config=config,
        log_file=log_file,
        rows_eval_break=rows_eval_break,
        log_progress=log_progress,
        use_progress_bar=use_progress_bar,
        async_processing=False,  # SummaC is sync only
        metric_title="SUMMAC CONVOLUTIONAL",
        pbar_desc="SummaC Conv",
        evaluator_factory=evaluator_factory,
        eval_method_name="evaluate_conv",
        output_column="summac_conv_score",
        logger=logger,
        max_concurrent=max_concurrent
    )


# =============================================================================
# Child Function 5: Default Classification Metrics
# =============================================================================

def evaluate_default_metrics(
    df,
    config,
    log_file="processing.log",
    rows_eval_break=None,
    log_progress=False,
    use_progress_bar=True,
    async_processing=False,  # Unused, kept for interface consistency
    max_concurrent=10  # Unused, kept for interface consistency
):
    """
    Evaluates default classification metrics (accuracy, F1, kappa).
    Compares annotation columns vs classification columns from taxonomy.
    Child function that specifies metric-specific parameters.

    Args:
        df (pd.DataFrame): Input DataFrame containing annotation and classification columns.
        config (Dict[str, Any]): Runtime settings.
        log_file (str): Path to the log file.
        rows_eval_break (Optional[int]): Stop after N rows (for testing).
        log_progress (bool): Enable verbose per-row logging.
        use_progress_bar (bool): Use tqdm progress bar.
        async_processing (bool): Unused, kept for interface consistency.

    Returns:
        pd.DataFrame: DataFrame with per-field match score columns added.
    """
    logger = setup_logger(log_file=log_file)
    logger.info("=" * 50)
    logger.info("DEFAULT METRICS EVALUATION")
    logger.info("=" * 50)

    # Create evaluator (loads taxonomy for field names)
    evaluator = DefaultMetricsEvaluator(config, logger)

    if not evaluator.field_names:
        logger.warning("No field names loaded from taxonomy. Skipping evaluation.")
        return df

    df_eval = df.copy()

    # Apply row limit if set (None = all rows)
    if rows_eval_break:
        df_eval = df_eval.head(rows_eval_break).copy()

    total_fields = len(evaluator.field_names)
    logger.info(f"Evaluating {total_fields} fields from taxonomy")
    logger.info(f"Rows to evaluate: {len(df_eval)}")

    # Evaluate each field from taxonomy
    with tqdm(total=total_fields, desc="Traditional Metrics", disable=not use_progress_bar) as pbar:
        for field_name in evaluator.field_names:
            annotation_col, classification_col = evaluator.get_column_pair(field_name)
            pbar.set_description(f"Field: {field_name}")

            # Skip if columns don't exist in DataFrame
            if annotation_col not in df_eval.columns or classification_col not in df_eval.columns:
                if log_progress:
                    logger.info(f"Skipping {field_name}: columns not found in DataFrame")
                pbar.update(1)
                continue

            # Collect per-row scores and labels for aggregation
            scores = []
            y_true = []
            y_pred = []

            for idx, row in df_eval.iterrows():
                ann = row[annotation_col]
                cls = row[classification_col]

                score = evaluator.evaluate_match(ann, cls)
                scores.append(score)

                # Accumulate for aggregate metrics
                y_true.append(str(ann).strip() if ann else '')
                y_pred.append(str(cls).strip() if cls else '')

            # Add per-row match column to DataFrame
            output_column = f"{field_name}_match"
            df_eval[output_column] = scores

            # Compute aggregate metrics per model
            if scores:
                for model_name in df_eval['model'].unique():
                    model_mask = df_eval['model'] == model_name
                    model_scores = df_eval.loc[model_mask, output_column].tolist()
                    model_y_true = df_eval.loc[model_mask, annotation_col].apply(
                        lambda x: str(x).strip() if x else '').tolist()
                    model_y_pred = df_eval.loc[model_mask, classification_col].apply(
                        lambda x: str(x).strip() if x else '').tolist()
                    
                    if model_scores:
                        accuracy = sum(model_scores) / len(model_scores)
                        labels = list(set(model_y_true + model_y_pred))
                        f1 = f1_score(model_y_true, model_y_pred, average='macro', zero_division=0)
                        kappa = cohen_kappa_score(model_y_true, model_y_pred, labels=labels)
                        logger.info(f"[{model_name}] {field_name}: Accuracy={accuracy:.3f} | F1={f1:.3f} | Kappa={kappa:.3f}")

            pbar.update(1)

    logger.info("=" * 50)
    logger.info("DEFAULT METRICS COMPLETE")
    logger.info("=" * 50)

    return df_eval


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main entry point for evaluation script.

    Loads configuration, reads output CSV, and executes evaluation pipeline.
    Runs enabled benchmarks based on config settings via registry.
    Saves results to configured output path.

    Returns:
        None
    """
    # Step 1: Load config (read once, use from RAM)
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger = setup_logger(log_file=config['logging']['file'])
    
    # Step 2: Load data
    input_path = config['paths']['output']['file']
    df = pd.read_csv(input_path, encoding='utf-8')
    output_file = config['paths'].get('eval_output', 'df_text_eval.csv')
    
    # Step 3: Read evaluation settings
    eval_config = config.get('evaluation', {})
    benchmarks = eval_config.get('benchmarks', {})
    
    # Step 4: Registry inside main - maps metric name to function
    metric_functions = {
        'geval_summarization': evaluate_geval_summarization,
        'geval_hallucination': evaluate_geval_hallucination,
        'summac_zs': evaluate_summac_zs,
        'summac_conv': evaluate_summac_conv,
        'default_metrics': evaluate_default_metrics,
    }
    
    # Step 5: Log enabled benchmarks
    logger.info("=" * 50)
    logger.info("EVALUATION BENCHMARKS")
    logger.info("=" * 50)
    
    df_eval = df.copy()
    
    # Step 6: Loop through enabled benchmarks, call via registry
    async_config = config.get('async', {})
    max_concurrent = async_config.get('max_concurrent_rows', 10)
    
    for key, enabled in benchmarks.items():
        if isinstance(enabled, bool) and enabled and key.startswith('enable_'):
            metric_name = key.replace('enable_', '')
            if metric_name in metric_functions:
                logger.info(f"{metric_name}: enabled, running...")
                func = metric_functions[metric_name]
                df_eval = func(
                    df=df_eval,
                    config=config,
                    log_file=config['logging']['file'],
                    rows_eval_break=eval_config.get('rows_eval_break'),
                    log_progress=config['logging'].get('log_progress', False),
                    use_progress_bar=config['display'].get('use_progress_bar', True),
                    async_processing=async_config.get('enabled', False),
                    max_concurrent=max_concurrent
                )
            else:
                logger.warning(f"{metric_name}: enabled but no function registered")
        elif isinstance(enabled, bool) and not enabled and key.startswith('enable_'):
            metric_name = key.replace('enable_', '')
            logger.info(f"{metric_name}: disabled")
    
    # Step 7: Save output
    df_eval.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved evaluation results to: {output_file}")
    logger.info("Evaluation complete.")


if __name__ == '__main__':
    main()
