'''
the orignal interface:
processed_df = process_dataframe_summary_and_classification(
    df=df_text,
    code_to_desc_map=code_to_desc_map,
    label_values_map=label_values_map,
    vllm_client=client,
    model_name=model,
    start_index=0,
    early_break=10,
    inner_loop_break=5,
    log_resources=False,
    log_prompts=False,
    log_response=False,
    log_progress=False
)
'''

# =============================================================================
# Step 0: Import modules
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import yaml
import json
import time
import gc
import asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_async
from tqdm.auto import tqdm

from src.processors import MessyTextProcessor, AsyncMessyTextProcessor
from src.utils import setup_logger, log_memory_usage, check_gpu_info, check_vllm_server


# =============================================================================
# Step 3: Processing function (Sync - Legacy Support)
# =============================================================================
def process_dataframe_summary_and_classification(
    df,
    code_to_desc_map,
    label_values_map,
    vllm_client,
    model_name,
    log_file="processing.log",
    start_index=0,
    early_break=None,
    inner_loop_break=None,
    log_resources=False,
    log_prompts=False,
    log_response=False,
    log_progress=False,
    use_progress_bar=True,
    max_tokens_summary=1024,
    max_tokens_classification=1048
):
    """
    Processes a DataFrame to generate summaries and classifications for text data using a synchronous approach.

    Maintains the same interface as the original notebook function for backward compatibility.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'text' column.
        code_to_desc_map (Dict[str, str]): Mapping of classification codes to descriptions.
        label_values_map (Dict[str, List[str]]): Mapping of classification codes to possible values.
        vllm_client (OpenAI): The synchronous OpenAI-compatible client.
        model_name (str): The name of the model to use.
        log_file (str): Path to the log file. Defaults to "processing.log".
        start_index (int): Row index to start processing from. Defaults to 0.
        early_break (Optional[int]): Stop after N rows (for testing). Defaults to None.
        inner_loop_break (Optional[int]): Stop classification after N keys per row (for testing). Defaults to None.
        log_resources (bool): Whether to log memory usage per row. Defaults to False.
        log_prompts (bool): Whether to log prompts (unused, internal implementation). Defaults to False.
        log_response (bool): Whether to log full responses (unused, internal implementation). Defaults to False.
        log_progress (bool): Enable verbose progress logging. Defaults to False.
        use_progress_bar (bool): Use tqdm progress bar (True) or log milestones (False). Defaults to True.
        max_tokens_summary (int): Max tokens for summary generation. Defaults to 1024.
        max_tokens_classification (int): Max tokens for classification generation. Defaults to 1048.

    Returns:
        pd.DataFrame: The processed DataFrame with new summary and classification columns.
    """
    # Setup logger from config
    logger = setup_logger(log_file=log_file)
    
    # Build config dict for processor
    config = {
        'model': {'name': model_name},
        'processing': {
            'temperature': 0.0,
            'max_tokens_summary': max_tokens_summary,
            'max_tokens_classification': max_tokens_classification
        }
    }
    
    # Build taxonomy dict for processor
    taxonomy = {
        'context_definitions': code_to_desc_map,
        'label_options': label_values_map
    }
    
    # Instantiate pure processor
    processor = MessyTextProcessor(vllm_client, config, taxonomy, logger)
    
    # Get classification keys, apply inner_loop_break
    all_keys = list(code_to_desc_map.keys())
    keys_to_classify = all_keys[:inner_loop_break] if inner_loop_break else all_keys
    
    # Initialize result columns
    df_processed = df.copy()
    new_columns = ['summary_all_context'] + [f'{key}_classification' for key in all_keys]
    for col in new_columns:
        if col not in df_processed.columns:
            df_processed[col] = ""
    
    # Prepare iteration
    results_list = []
    df_to_process = df_processed.iloc[start_index:]
    total_rows = len(df_to_process)
    if early_break is not None and early_break < total_rows:
        total_rows = early_break
    
    row_counter = 0
    start_time = time.time()
    logged_milestones = set()
    
    # Outer loop: DataFrame rows
    with tqdm(total=total_rows, desc="Processing", position=0, leave=True, 
              disable=not use_progress_bar) as pbar:
        for row in df_to_process.itertuples():
            # Early break check BEFORE processing (handles early_break=0 correctly)
            if early_break is not None and row_counter >= early_break:
                break
            
            pbar.set_description(f"Processing (Index: {row.Index})")
            
            if log_progress:
                logger.info(f"Processing row {row.Index}")
            
            current_row_results = {'index': row.Index}
            
            # Step 1: Clean text
            text = str(row.text) if hasattr(row, 'text') else ""
            cleaned_text = processor.clean_text(text)
            
            # Step 2: Summarize
            if cleaned_text.strip():
                summary = processor.summarize_text(cleaned_text)
            else:
                summary = 'No relevant information found'
            
            current_row_results['summary_all_context'] = summary
            
            if log_response:
                logger.info(f"Summary for {row.Index}: {summary[:100]}...")
            
            # Step 3: Classify (inner loop)
            if summary and summary != 'No relevant information found':
                with tqdm(keys_to_classify, total=len(keys_to_classify), 
                         desc="Classifying", leave=False, position=1,
                         disable=not use_progress_bar) as pbar_inner:
                    for key in pbar_inner:
                        classification = processor.classify_summary(summary, key)
                        current_row_results[f'{key}_classification'] = classification
                        
                        if log_response:
                            logger.info(f"  {key}: {classification}")
            
            # Fill remaining keys with empty if inner_loop_break was applied
            for key in all_keys:
                if f'{key}_classification' not in current_row_results:
                    current_row_results[f'{key}_classification'] = ""
            
            results_list.append(current_row_results)
            
            # Memory logging
            if log_resources:
                log_memory_usage(logger, f"After row {row.Index}")
            
            # Cleanup
            del cleaned_text, summary
            gc.collect()
            
            row_counter += 1
            pbar.update(1)
            
            # Log progress at milestones when progress bar disabled (server mode)
            if not use_progress_bar and total_rows > 0:
                pct = int((row_counter / total_rows) * 100)
                # Log at 10% milestones
                milestone = (pct // 10) * 10
                if milestone > 0 and milestone not in logged_milestones:
                    elapsed = time.time() - start_time
                    rate = row_counter / elapsed if elapsed > 0 else 0
                    eta = (total_rows - row_counter) / rate if rate > 0 else 0
                    logger.info(f"Index {row.Index} | {milestone}% ({row_counter}/{total_rows}) | "
                               f"Elapsed: {elapsed:.0f}s | {rate:.2f} rows/sec | ETA: {eta:.0f}s")
                    logged_milestones.add(milestone)
            
            # Throttle every 5 rows
            if row_counter % 5 == 0:
                time.sleep(2)
    
    # Update DataFrame with results
    if results_list:
        results_df = pd.DataFrame(results_list).set_index('index')
        df_processed.update(results_df)
    
    logger.info(f"Processing complete. Processed {row_counter} rows.")
    return df_processed


# =============================================================================
# Step 3b: Processing function (Async - High Performance)
# =============================================================================
async def process_row_async(row, processor, keys_to_classify, semaphore, all_keys):
    """
    Processes a single row asynchronously, including cleaning, summarization, and classification.

    Args:
        row (NamedTuple): A pandas row namedtuple containing the 'text' and 'Index'.
        processor (AsyncMessyTextProcessor): The async processor instance.
        keys_to_classify (List[str]): List of keys to classify for this row.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent execution.
        all_keys (List[str]): Complete list of keys to ensure all columns are present in output.

    Returns:
        Dict[str, Any]: A dictionary containing the processing results for the row.
    """
    async with semaphore:
        current_row_results = {'index': row.Index}
        
        # Step 1: Clean text
        text = str(row.text) if hasattr(row, 'text') else ""
        cleaned_text = processor.clean_text(text)
        
        # Step 2: Summarize
        if cleaned_text.strip():
            summary = await processor.summarize_text(cleaned_text)
        else:
            summary = 'No relevant information found'
        
        current_row_results['summary_all_context'] = summary
        
        # Step 3: Classify (Parallel inner loop)
        if summary and summary != 'No relevant information found':
            # Create tasks for all keys to run in parallel
            tasks = []
            for key in keys_to_classify:
                tasks.append(processor.classify_summary(summary, key))
            
            # Run all classifications concurrently
            results = await asyncio.gather(*tasks)
            
            # Store results
            for key, classification in zip(keys_to_classify, results):
                current_row_results[f'{key}_classification'] = classification
        
        # Fill remaining keys or if no summary found
        for key in all_keys:
            if f'{key}_classification' not in current_row_results:
                current_row_results[f'{key}_classification'] = ""
                
        return current_row_results

async def process_dataframe_async(
    df,
    code_to_desc_map,
    label_values_map,
    vllm_client,
    model_name,
    log_file="processing.log",
    start_index=0,
    early_break=None,
    inner_loop_break=None,
    log_resources=False,
    log_prompts=False,  # Kept for interface compatibility, unused
    log_response=False, # Kept for interface compatibility, unused
    log_progress=False, # Kept for interface compatibility, unused
    use_progress_bar=True,
    max_concurrent_rows=50,
    max_tokens_summary=1024,
    max_tokens_classification=1048
):
    """
    Processes a DataFrame asynchronously using concurrent execution for high throughput.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'text' column.
        code_to_desc_map (Dict[str, str]): Mapping of classification codes to descriptions.
        label_values_map (Dict[str, List[str]]): Mapping of classification codes to possible values.
        vllm_client (AsyncOpenAI): The asynchronous OpenAI-compatible client.
        model_name (str): The name of the model to use.
        log_file (str): Path to the log file. Defaults to "processing.log".
        start_index (int): Row index to start processing from. Defaults to 0.
        early_break (Optional[int]): Stop after N rows (for testing). Defaults to None.
        inner_loop_break (Optional[int]): Stop classification after N keys per row (for testing). Defaults to None.
        log_resources (bool): Whether to log memory usage per row. Defaults to False.
        log_prompts (bool): Unused, kept for compatibility. Defaults to False.
        log_response (bool): Unused, kept for compatibility. Defaults to False.
        log_progress (bool): Unused, kept for compatibility. Defaults to False.
        use_progress_bar (bool): Use tqdm progress bar. Defaults to True.
        max_concurrent_rows (int): Maximum number of concurrent row processing tasks. Defaults to 50.
        max_tokens_summary (int): Max tokens for summary generation. Defaults to 1024.
        max_tokens_classification (int): Max tokens for classification generation. Defaults to 1048.

    Returns:
        pd.DataFrame: The processed DataFrame with new summary and classification columns.
    """
    # Setup logger from config
    logger = setup_logger(log_file=log_file)
    
    config = {
        'model': {'name': model_name},
        'processing': {
            'temperature': 0.0,
            'max_tokens_summary': max_tokens_summary,
            'max_tokens_classification': max_tokens_classification
        }
    }
    
    taxonomy = {
        'context_definitions': code_to_desc_map,
        'label_options': label_values_map
    }
    
    # Instantiate Async processor
    processor = AsyncMessyTextProcessor(vllm_client, config, taxonomy, logger)
    
    all_keys = list(code_to_desc_map.keys())
    keys_to_classify = all_keys[:inner_loop_break] if inner_loop_break else all_keys
    
    # Initialize columns in input df (optional, for structure)
    df_processed = df.copy()
    new_columns = ['summary_all_context'] + [f'{key}_classification' for key in all_keys]
    for col in new_columns:
        if col not in df_processed.columns:
            df_processed[col] = ""
            
    df_to_process = df_processed.iloc[start_index:]
    total_rows = len(df_to_process)
    if early_break is not None and early_break < total_rows:
        df_to_process = df_to_process.iloc[:early_break]
        total_rows = early_break

    # Concurrency Control
    semaphore = asyncio.Semaphore(max_concurrent_rows)
    
    logger.info(f"Starting async processing with {max_concurrent_rows} concurrent rows.")
    
    # Create tasks
    tasks = []
    for row in df_to_process.itertuples():
        task = process_row_async(row, processor, keys_to_classify, semaphore, all_keys)
        tasks.append(task)
    
    # Run tasks with progress bar
    results_list = []
    for f in tqdm_async.as_completed(tasks, total=len(tasks), desc="Processing Rows", disable=not use_progress_bar):
        result = await f
        results_list.append(result)
        if log_resources and len(results_list) % 100 == 0:
             log_memory_usage(logger, f"Processed {len(results_list)} rows")

    # Update DataFrame
    if results_list:
        results_df = pd.DataFrame(results_list).set_index('index')
        df_processed.update(results_df)
        
    logger.info(f"Processing complete. Processed {len(results_list)} rows.")
    return df_processed


# =============================================================================
# Step 4: Main function
# =============================================================================
def main():
    """
    Main entry point for the script.

    Loads configuration, initializes clients, performs pre-flight checks, and executes the processing pipeline.
    Supports both synchronous and asynchronous modes based on configuration.

    Returns:
        None
    """
    # Step 1: Load config (first, so logger can use config)
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    
    logger = setup_logger(log_file=settings['logging']['file'])
    
    # Step 2: Load data
    df_text = pd.read_csv(settings['paths']['input'], encoding='utf-8')
    
    with open('config/taxonomy.json', 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    code_to_desc_map = taxonomy['context_definitions']
    label_values_map = taxonomy['label_options']
    
    # Step 3a: Create SYNC client for Pre-flight checks
    sync_client = OpenAI(
        base_url=settings['model']['api_base'],
        api_key=settings['model']['api_key']
    )
    
    logger.info("=" * 50)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 50)
    
    # GPU check
    gpu_info = check_gpu_info(logger)
    if gpu_info is None:
        logger.error("No GPU available. Exiting.")
        sys.exit(1)
    
    # vLLM server check
    vllm_ok, available_models, test_result = check_vllm_server(
        sync_client,
        settings['model']['name'],
        logger
    )
    
    if not vllm_ok:
        logger.error(f"Pre-flight FAILED. Models found: {available_models}")
        sys.exit(1)
    
    logger.info("Pre-flight checks PASSED")
    logger.info("=" * 50)
    
    # Step 3b: Select Processing Mode
    async_processing = settings['async'].get('enabled', True)
    
    if async_processing:
        logger.info("Using ASYNC processing mode.")
        max_retries = settings['async'].get('max_retries', 2)
        max_concurrent_rows = settings['async'].get('max_concurrent_rows', 50)
        
        async_client = AsyncOpenAI(
            base_url=settings['model']['api_base'],
            api_key=settings['model']['api_key'],
            max_retries=max_retries
        )
        
        processed_df = asyncio.run(process_dataframe_async(
            df=df_text,
            code_to_desc_map=code_to_desc_map,
            label_values_map=label_values_map,
            vllm_client=async_client,
            model_name=settings['model']['name'],
            log_file=settings['logging']['file'],
            start_index=settings['processing']['start_index'],
            early_break=settings['processing']['early_break'],
            inner_loop_break=settings['processing']['inner_loop_break'],
            log_resources=settings['logging']['log_resources'],
            log_prompts=settings['logging']['log_prompts'],
            log_response=settings['logging']['log_response'],
            log_progress=settings['logging']['log_progress'],
            use_progress_bar=settings['display']['use_progress_bar'],
            max_concurrent_rows=max_concurrent_rows,
            max_tokens_summary=settings['processing'].get('max_tokens_summary', 1024),
            max_tokens_classification=settings['processing'].get('max_tokens_classification', 1048)
        ))
    else:
        logger.info("Using SYNC processing mode (Legacy).")
        processed_df = process_dataframe_summary_and_classification(
            df=df_text,
            code_to_desc_map=code_to_desc_map,
            label_values_map=label_values_map,
            vllm_client=sync_client,  # Reuse the sync client
            model_name=settings['model']['name'],
            log_file=settings['logging']['file'],
            start_index=settings['processing']['start_index'],
            early_break=settings['processing']['early_break'],
            inner_loop_break=settings['processing']['inner_loop_break'],
            log_resources=settings['logging']['log_resources'],
            log_prompts=settings['logging']['log_prompts'],
            log_response=settings['logging']['log_response'],
            log_progress=settings['logging']['log_progress'],
            use_progress_bar=settings['display']['use_progress_bar'],
            max_tokens_summary=settings['processing'].get('max_tokens_summary', 1024),
            max_tokens_classification=settings['processing'].get('max_tokens_classification', 1048)
        )
    
    # Save output
    processed_df.replace(['No information', 'No relevant information found'], '', inplace=True)
    output_path = settings['paths']['output']
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()
