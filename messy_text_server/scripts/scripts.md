## scripts/ docstring index


### module: <module>

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

### function: _process_dataframe_classification_sync

    """
    Synchronous classification-only pipeline.

    Mirrors the classification part of process_dataframe_summary_and_classification
    from run_processing.py, but:
      - Assumes 'summary_all_context' is already populated.
      - Does not call summarize_text.
    """

### asyncfunction: _classify_row_async

    """
    Asynchronous per-row classification helper.

    Mirrors the classification part of process_row_async in run_processing.py,
    but uses an existing summary_all_context instead of recomputing summaries.
    """

### asyncfunction: _process_dataframe_classification_async

    """
    Asynchronous classification-only pipeline.

    Reuses AsyncMessyTextProcessor.classify_summary with the same taxonomy and
    prompts as run_processing.py, but assumes 'summary_all_context' is present.
    """

### function: main

    """
    Main entry point for non-conversational classification-only processing.

    Behavior mirrors run_processing.py:
        - Reads config/settings.yaml
        - Runs GPU + vLLM pre-flight checks
        - Chooses async vs sync based on async.enabled
        - Writes classification columns based on 'summary_all_context'
    """


### module: <module>

"""
Evaluation script for summarization quality.
Reads output CSV from config and evaluates summary quality using registered metrics.
Outputs results with all original columns plus evaluation scores.
"""

### function: _ensure_nltk_punkt_tab

    """
    Ensures NLTK punkt_tab is available for SummaC sentence tokenization.

    Behavior:
        1) Check for local punkt_tab data.
        2) If missing and auto_download=True, attempt download once.
        3) If still missing, fail fast with a reproducible command.
    """

### asyncfunction: _evaluate_single_row_async

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

### asyncfunction: _evaluate_metric_async

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

### function: _evaluate_metric

    """
    Parent helper function containing shared evaluation logic.
    Called by child functions that specify the metric-specific parameters.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'text' and 'summary_all_context' columns.
        config (Dict[str, Any]): Runtime settings.
        log_file (str): Path to the log file.
        rows_eval_limit (Optional[int]): Stop after N rows (for testing).
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

### function: evaluate_geval_summarization

    """
    Evaluates summarization quality using G-Eval.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'geval_summarization_score' column.
    """

### function: evaluate_geval_hallucination

    """
    Evaluates hallucination using G-Eval.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'geval_hallucination_score' column.
    """

### function: evaluate_summac_zs

    """
    Evaluates factual consistency using SummaC Zero-Shot.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'summac_zs_score' column.
    """

### function: evaluate_summac_conv

    """
    Evaluates factual consistency using SummaC Convolutional.
    Child function that specifies metric-specific parameters.

    Returns:
        pd.DataFrame: DataFrame with 'summac_conv_score' column.
    """

### function: evaluate_default_metrics

    """
    Evaluates default classification metrics (accuracy, F1, kappa).
    Compares annotation columns vs classification columns from taxonomy.
    Child function that specifies metric-specific parameters.

    Args:
        df (pd.DataFrame): Input DataFrame containing annotation and classification columns.
        config (Dict[str, Any]): Runtime settings.
        log_file (str): Path to the log file.
        rows_eval_limit (Optional[int]): Stop after N rows (for testing).
        log_progress (bool): Enable verbose per-row logging.
        use_progress_bar (bool): Use tqdm progress bar.
        async_processing (bool): Unused, kept for interface consistency.

    Returns:
        pd.DataFrame: DataFrame with per-field match score columns added.
    """

### function: main

    """
    Main entry point for evaluation script.

    Loads configuration, reads output CSV, and executes evaluation pipeline.
    Runs enabled benchmarks based on config settings via registry.
    Saves results to configured output path.

    Returns:
        None
    """


### module: <module>

"""
Standalone summarization entry point for MessyText (non-conversational).

This script reuses the existing MessyTextProcessor and configuration logic
from run_processing.py, but only performs:

    text -> cleaned_text -> summary_all_context

No classification calls are made here. The intent is:

- run_processing.py      : summary + classification in one pass
- run_summary.py         : summary only
- run_classification.py  : classification only (on precomputed summaries)
"""

### function: _process_dataframe_summary_sync

    """
    Synchronous summarization-only pipeline.

    Mirrors the summarization part of process_dataframe_summary_and_classification
    from run_processing.py but omits any classification logic. All core LLM
    behavior is delegated to MessyTextProcessor.summarize_text, so no prompt
    or parsing logic is duplicated here.
    """

### asyncfunction: _summarize_row_async

    """
    Asynchronous per-row summarization helper.

    This mirrors the summarization part of process_row_async in run_processing.py
    but does not invoke any classification calls.
    """

### asyncfunction: _process_dataframe_summary_async

    """
    Asynchronous summarization-only pipeline.

    This reuses AsyncMessyTextProcessor.summarize_text and the same taxonomy,
    prompts, and configuration shape as run_processing.py while omitting
    classification.
    """

### function: main

    """
    Main entry point for non-conversational summarization-only processing.

    Behavior mirrors run_processing.py:
        - Reads config/settings.yaml
        - Runs GPU + vLLM pre-flight checks
        - Chooses async vs sync based on async.enabled
        - Writes a DataFrame with 'summary_all_context' populated
    """


### module: <module>

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

### asyncfunction: _process_victim_async

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

### function: _process_victim_sync

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

### asyncfunction: _process_dataframe_conversation_async

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

### function: _process_dataframe_conversation_sync

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

### function: main

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
