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
    show_resources=False,
    print_prompts=False,
    print_response=False,
    print_progress=False
)
'''

# =============================================================================
# Step 0: Import modules
# =============================================================================
import sys
import pandas as pd
import yaml
import json
import time
import gc
from openai import OpenAI
from tqdm.auto import tqdm

from src.messy_text_processor import MessyTextProcessor
from src.utils import setup_logger, log_memory_usage, check_gpu_info, check_vllm_server


# =============================================================================
# Step 3: Processing function (policy/orchestration layer)
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
    show_resources=False,
    print_prompts=False,
    print_response=False,
    print_progress=False,
    use_progress_bar=True
):
    """
    Processes a DataFrame to generate summaries and classifications for text data.
    Maintains same interface as original notebook function.
    
    Args:
        df: Input DataFrame with 'text' column.
        code_to_desc_map: Dict mapping classification codes to descriptions.
        label_values_map: Dict mapping classification codes to possible values.
        vllm_client: OpenAI-compatible client for vLLM.
        model_name: Model name string.
        log_file: Path to log file.
        start_index: Row index to start processing from.
        early_break: Stop after N rows (for testing).
        inner_loop_break: Stop classification after N keys per row (for testing).
        show_resources: Log memory usage per row.
        print_prompts: Log prompts (not implemented - prompts are internal).
        print_response: Log responses (not implemented - responses are internal).
        print_progress: Enable verbose progress logging.
        use_progress_bar: Use tqdm progress bar (True) or log milestones (False for server).
    
    Returns:
        pd.DataFrame: Processed DataFrame with summary and classification columns.
    """
    # Setup logger from config
    logger = setup_logger(log_file=log_file)
    
    # Build config dict for processor
    config = {
        'model': {'name': model_name},
        'processing': {
            'temperature': 0.0,
            'max_tokens_summary': 1024,
            'max_tokens_classification': 1048
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
            
            if print_progress:
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
            
            if print_response:
                logger.info(f"Summary for {row.Index}: {summary[:100]}...")
            
            # Step 3: Classify (inner loop)
            if summary and summary != 'No relevant information found':
                with tqdm(keys_to_classify, total=len(keys_to_classify), 
                         desc="Classifying", leave=False, position=1,
                         disable=not use_progress_bar) as pbar_inner:
                    for key in pbar_inner:
                        classification = processor.classify_summary(summary, key)
                        current_row_results[f'{key}_classification'] = classification
                        
                        if print_response:
                            logger.info(f"  {key}: {classification}")
            
            # Fill remaining keys with empty if inner_loop_break was applied
            for key in all_keys:
                if f'{key}_classification' not in current_row_results:
                    current_row_results[f'{key}_classification'] = ""
            
            results_list.append(current_row_results)
            
            # Memory logging
            if show_resources:
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
# Step 4: Main function
# =============================================================================
def main():
    # Step 1: Load config (first, so logger can use config)
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    
    # Setup logger from config
    logger = setup_logger(log_file=settings['logging']['file'])
    
    # Step 2: Load data
    df_text = pd.read_csv('df_text.csv', encoding='utf-8')
    
    with open('config/taxonomy.json', 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    
    code_to_desc_map = taxonomy['context_definitions']
    label_values_map = taxonomy['label_options']
    
    # Step 3a: Create client
    client = OpenAI(
        base_url=settings['model']['api_base'],
        api_key=settings['model']['api_key']
    )
    
    # Pre-flight checks
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
        client,
        settings['model']['name'],
        logger
    )
    
    if not vllm_ok:
        logger.error(f"Pre-flight FAILED. Models found: {available_models}")
        sys.exit(1)
    
    logger.info("Pre-flight checks PASSED")
    logger.info("=" * 50)
    
    # Step 3b: Call processing function (all parameters from config)
    processed_df = process_dataframe_summary_and_classification(
        df=df_text,
        code_to_desc_map=code_to_desc_map,
        label_values_map=label_values_map,
        vllm_client=client,
        model_name=settings['model']['name'],
        log_file=settings['logging']['file'],
        start_index=settings['processing']['start_index'],
        early_break=settings['processing']['early_break'],
        inner_loop_break=settings['processing']['inner_loop_break'],
        show_resources=settings['processing']['show_resources'],
        print_prompts=settings['processing']['print_prompts'],
        print_response=settings['processing']['print_response'],
        print_progress=settings['processing']['print_progress'],
        use_progress_bar=settings['processing']['use_progress_bar']
    )
    
    # Save output
    processed_df.replace(['No information', 'No relevant information found'], '', inplace=True)
    output_path = settings['output']['path']
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Output saved to {output_path}")


if __name__ == '__main__':
    main()
