## src/ docstring index


### class: MessyTextLogicMixin

    """
    Contains pure logic for MessyTextProcessor.
    Handles text cleaning, prompt construction, and response parsing.
    Decoupled from I/O (client calls).
    """

### class: MessyTextProcessor

    """
    Synchronous processor implementation.
    Maintains exact backward compatibility with original implementation.
    """

### class: AsyncMessyTextProcessor

    """
    Asynchronous processor implementation for high-throughput servers.
    Uses async/await patterns with AsyncOpenAI client.
    """


### module: <module>

"""
Why not using comet or bleu to evaluate the model? They are for translation tasks.
Why not using rouge or similar benchmarks to evaluate the model? They requires human reference.
QAFactEval or G-Eval works, but they are QA based. 
We also use regular benchmarks like accuracy, precision, recall, f1-score, etc.

adapted QAFactEval:
They use: QA from sources vs QA from summaries, and get LERC scores.
We use: Machine labelling from sources vs machine labelling from summaries, and get LERC scores.

G-Eval:
Uses DeepEval's built-in SummarizationMetric and HallucinationMetric with local vLLM model.
Reference: https://github.com/confident-ai/deepeval
"""

### class: VLLMModelLogicMixin

    """
    Pure logic mixin for VLLMModel classes.
    Provides shared model configuration and response parsing logic.
    Decoupled from I/O (client calls) for sync/async flexibility.

    Attributes:
        _model_name (str): The full model name for vLLM inference.
        temperature (float): Generation temperature parameter.

    Methods:
        model_name: Property returning the model name.
        get_model_name(): Returns formatted model name for display.
        _parse_response(): Parses raw model response, optionally validating schema.
    """

### class: VLLMModel

    """
    Synchronous DeepEval model wrapper for vLLM.
    Implements DeepEvalBaseLLM interface using synchronous OpenAI client.
    Preserves full model name without prefix stripping.

    Attributes:
        client (OpenAI): The synchronous OpenAI-compatible client instance.

    Methods:
        load_model(): Returns the OpenAI client.
        generate(): Synchronous text generation with optional schema validation.
        a_generate(): Async fallback (calls sync generate for compatibility).
    """

### class: AsyncVLLMModel

    """
    Asynchronous DeepEval model wrapper for vLLM.
    Implements DeepEvalBaseLLM interface using asynchronous AsyncOpenAI client.
    Provides non-blocking operations for high-throughput server environments.

    Attributes:
        client (AsyncOpenAI): The asynchronous OpenAI-compatible client instance.

    Methods:
        load_model(): Returns the AsyncOpenAI client.
        generate(): Raises RuntimeError (sync not supported).
        a_generate(): Asynchronous text generation with optional schema validation.
    """

### class: GEvalEvaluatorLogicMixin

    """
    Pure logic mixin for GEvalEvaluator classes.
    Provides shared metric and test case creation logic.
    Decoupled from I/O (model calls) for sync/async flexibility.

    Attributes:
        config (Dict[str, Any]): Runtime settings containing model configuration.
        logger (logging.Logger): Logger instance for error/info messages.

    Methods:
        _create_summarization_metric(): Creates SummarizationMetric instance.
        _create_hallucination_metric(): Creates HallucinationMetric instance.
        _create_summarization_test_case(): Creates test case for summarization.
        _create_hallucination_test_case(): Creates test case for hallucination.
    """

### class: GEvalEvaluator

    """
    Synchronous evaluator for summary quality using DeepEval metrics.
    Uses local vLLM model via synchronous OpenAI-compatible client.
    Suitable for local development and single-threaded environments.

    Attributes:
        model (VLLMModel): The synchronous vLLM model wrapper.

    Methods:
        evaluate_summarization(): Evaluates summary quality (sync).
        evaluate_hallucination(): Evaluates hallucination level (sync).
    """

### class: AsyncGEvalEvaluator

    """
    Asynchronous evaluator for summary quality using DeepEval metrics.
    Uses local vLLM model via asynchronous AsyncOpenAI client.
    Suitable for high-throughput server environments with non-blocking I/O.

    Attributes:
        model (AsyncVLLMModel): The asynchronous vLLM model wrapper.

    Methods:
        evaluate_summarization(): Evaluates summary quality (async).
        evaluate_hallucination(): Evaluates hallucination level (async).
    """

### class: SummaCEvaluator

    """
    Evaluator for factual consistency using SummaC NLI-based models.
    Provides both zero-shot (SummaCZS) and convolutional (SummaCConv) metrics.
    Reference: https://github.com/tingofurro/summac

    SummaC measures factual consistency by checking if summary sentences
    are entailed by the source document using Natural Language Inference.

    Attributes:
        config (Dict[str, Any]): Runtime settings containing 'summac' configuration.
        logger (logging.Logger): Logger instance for error/info messages.
        model_zs (SummaCZS): Zero-shot SummaC model instance.
        model_conv (SummaCConv): Convolutional SummaC model instance.

    Methods:
        evaluate_zs(): Evaluates consistency using SummaCZS (zero-shot).
        evaluate_conv(): Evaluates consistency using SummaCConv (learned aggregation).
    """

### class: DefaultMetricsEvaluator

    """
    Evaluator for default classification metrics.
    Compares annotation (ground truth) vs classification (prediction) per row.
    Reads taxonomy.json to determine valid field names and column mappings.

    Per-row method returns match indicator (1.0 if match, 0.0 otherwise).
    Aggregate metrics (accuracy, F1, kappa) are computed by orchestration layer.

    Attributes:
        config (Dict[str, Any]): Runtime settings containing paths.taxonomy.
        logger (logging.Logger): Logger instance for error/info messages.
        field_names (List[str]): Valid field names from taxonomy.

    Methods:
        evaluate_match(): Per-row match evaluation (1.0 or 0.0).
        get_column_pair(): Returns (annotation_col, classification_col) for a field.
    """


### class: ProcessorResult

    """
    Generic result container object for a single LLM-backed processor call.

    This object is intentionally task-agnostic: it can represent summary,
    classification, or any future task that uses the guided_json interface.

    Attributes:
        task_name: Logical name of the task, e.g. "summary", "classification".
        model_name: Identifier of the model that produced this result.
        declared_fields: Schema of expected output fields, taken from the
            guided_json properties (field name -> property spec).
        values: Parsed values for the declared_fields, extracted from the
            model's JSON output.
        input_text: Raw text sent to the model (messages[0].content).
        input_struct: Best-effort parsed structure of the input prompt
            (after repair_json), or None if parsing failed.
        output_text: Raw text returned by the model.
        output_struct: Parsed JSON output (after repair_json), or None if
            parsing failed.
        doc_id: Optional identifier for the logical unit of work (row index,
            document id, victim id, etc.).
        error: Optional error message captured during parsing or construction.
        metadata: Free-form dictionary for additional audit fields (timings,
            request ids, etc.).
    """

### class: MessyTextLogicMixin

    """
    Contains pure logic for MessyTextProcessor.
    Handles text cleaning, prompt construction, and response parsing.
    Decoupled from I/O (client calls).
    """

### function: _attach_span_offsets_to_result

    """
    Enrich summary_by_item entries with offsets computed from source_text.

    Offsets are only computed when summary_by_item is a dict; otherwise the
    result is returned unchanged.
    """

### class: MessyTextProcessor

    """
    Synchronous processor implementation.
    Maintains exact backward compatibility with original implementation.
    """

### class: AsyncMessyTextProcessor

    """
    Asynchronous processor implementation for high-throughput servers.
    Uses async/await patterns with AsyncOpenAI client.
    """

### class: MessyTextConversationState

    """
    Holds state for a multi-turn MessyText conversation.

    This object tracks all results across turns, enabling traceback to original
    spans and document sources. It provides properties for convenient access to
    the most recent result and summary.

    Attributes:
        turn_index (int): Number of turns that have been processed so far.
        results (List[ProcessorResult]): All structured results from each turn,
            preserving the full conversation history including spans and doc_ids.
    """

### class: MessyTextConversationTurnProcessor

    """
    Performs a single sequential summarization turn with optional conversation state.

    This class wraps a synchronous MessyTextProcessor and provides a higher-level
    interface that:

    1. Accepts raw input text and an optional MessyTextConversationState.
    2. Cleans the text using the shared regex logic.
    3. Calls summarize_text to obtain a per-turn summary (using the current
       prompt design as a placeholder for future, memory-aware prompts).
    4. Returns both the summary for this turn and an updated conversation state.

    The current implementation does not yet inject the previous state into the
    underlying prompt. That behavior can be added later by updating the logic in
    this class and/or _get_summary_args without changing the public interface.
    """

### class: MessyTextConversationOrchestrator

    """
    Orchestrates a multi-turn MessyText conversation over a sequence of documents.

    This class is responsible for:

    - Deciding where to start (initial state).
    - Stepping through documents sequentially.
    - Passing the updated MessyTextConversationState between turns.
    - Applying optional stopping criteria (e.g., maximum turns or a custom
      stop condition callback).

    It does not perform any I/O or DataFrame operations; callers are expected
    to adapt their own data structures (e.g., lists of strings, DataFrame rows)
    to the `texts` iterable interface.
    """

### class: AsyncMessyTextConversationTurnProcessor

    """
    Asynchronous counterpart of MessyTextConversationTurnProcessor.

    This class wraps an AsyncMessyTextProcessor and exposes an async interface
    for running a single summarization turn with optional conversation state.
    """

### class: AsyncMessyTextConversationOrchestrator

    """
    Asynchronous orchestrator for multi-turn MessyText conversations.

    This class mirrors MessyTextConversationOrchestrator but exposes an async
    run_conversation method so that multiple conversations can be executed
    concurrently by higher-level pipelines.
    """

### class: LabelExtractorLogicMixin

    """
    Contains pure logic for single-label span extraction from document text.

    Handles text cleaning and prompt construction for extracting spans
    relevant to exactly one taxonomy label from a single document.
    Decoupled from I/O (client calls).

    This mixin reads prompt configuration from the following keys:

        prompts.label_extract_first  — first-turn extraction (no previous spans).
        prompts.label_extract_update — update-turn extraction (has previous spans).

    These are independent of the prompt keys used by the existing
    MessyTextLogicMixin so that old prompts remain untouched.
    """

### class: LabelExtractor

    """
    Synchronous single-label extraction processor.

    Processes one taxonomy label against one document text, returning a
    ProcessorResult with the extraction output for that label.
    """

### class: AsyncLabelExtractor

    """
    Asynchronous single-label extraction processor.

    Processes one taxonomy label against one document text, returning a
    ProcessorResult with the extraction output for that label.
    Uses async/await patterns with AsyncOpenAI client.
    """

### class: LabelSummaryLogicMixin

    """
    Contains pure logic for producing a document summary from per-label
    extraction results.

    Handles prompt construction for the summarization call that takes
    pre-extracted label evidence as input and produces a coherent
    document-level summary.
    Decoupled from I/O (client calls).

    This mixin reads prompt configuration from the following key:

        prompts.label_summary — summarization from per-label evidence.

    This is independent of the prompt keys used by the existing
    MessyTextLogicMixin so that old prompts remain untouched.
    """

### class: LabelSummaryProcessor

    """
    Synchronous label-based summary processor.

    Takes pre-extracted per-label ProcessorResults and the original document
    text, makes a single LLM call, and returns a ProcessorResult whose values
    match the existing conversation summary contract (info_found,
    relevant_context, summary_by_item, summary).
    """

### class: AsyncLabelSummaryProcessor

    """
    Asynchronous label-based summary processor.

    Takes pre-extracted per-label ProcessorResults and the original document
    text, makes a single LLM call, and returns a ProcessorResult whose values
    match the existing conversation summary contract (info_found,
    relevant_context, summary_by_item, summary).
    Uses async/await patterns with AsyncOpenAI client.
    """


### function: serialize_result_entry

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

### function: serialize_state_entry

    """
    Convert a MessyTextConversationState into a single record for storage.

    Args:
        state: Conversation state holding all ProcessorResult objects.
        victim_id: Logical victim identifier.
        model_name: Name of the model used during processing.

    Returns:
        Dict with victim/model identifiers, turn count, and serialized results.
    """

### function: flatten_spans_from_state

    """
    Flatten summary_by_item spans from a conversation state into row records.

    Args:
        state: Conversation state containing ProcessorResult objects.
        victim_id: Logical victim identifier.
        model_name: Name of the model used during processing.

    Returns:
        List of dict rows suitable for CSV export, one row per span.
    """

### function: _write_csv_with_extend

    """
    Write rows to CSV, optionally extending existing content while replacing
    rows for the current model.
    """

### function: write_results

    """Persist flattened result rows."""

### function: _expand_values

    """
    Expand a 'values' column (dict or JSON string) into individual columns.

    If the column is absent, the DataFrame is returned unchanged.
    """

### function: write_states

    """Persist serialized conversation states."""

### function: write_spans

    """Persist flattened spans."""


### function: setup_logger

    """
    Sets up a file-based logger for long-running server jobs.
    Also configures httpx logger to expose retry attempts.
    """

### function: log_memory_usage

    """
    Logs current memory usage of the process.
    Replaces the notebook's 'show_resource' with system-level tracking.
    """

### function: get_deep_size

    """
    Recursively calculates object size (from notebook logic).
    """

### function: is_informative_summary

    """
    Determines whether a summary string should be treated as containing
    victim-relevant information, as opposed to a generic "no information"
    placeholder.

    This helper centralizes the normalization and sentinel matching logic so
    that multi-turn runners can make consistent decisions about when to update
    the running summary.

    Args:
        summary (Optional[str]): The summary text returned by the model.

    Returns:
        bool: True if the summary is informative, False if it represents a
        no-information result.
    """

### function: check_gpu_info

    """
    Get actual GPU hardware info via nvidia-smi.
    Returns list of GPU info dicts or None if no GPU.
    """

### function: check_vllm_server

    """
    Check vLLM server connectivity and available models.
    Matches notebook pattern: list models + test request.
    
    Returns:
        tuple: (success: bool, available_models: list, test_result: str or None)
    """

### function: compute_span_offsets

    """
    Compute character offsets for extracted spans by searching in source documents.

    This is a post-hoc utility that enriches summary_by_item with offset information
    for traceback to original text positions.

    Args:
        summary_by_item (dict): Dictionary mapping label keys to lists of span items.
            Expected format: {
                "<label_key>": [
                    {"span": "<exact text>", "doc_id": "<id>"},
                    ...
                ]
            }
        get_source_text (callable): Function that takes a doc_id and returns the
            source text string. Signature: (doc_id: Any) -> str

    Returns:
        dict: The same structure with 'offset' added to each span item.
            Offset is -1 if span not found in source text.

    Example:
        >>> def get_source(doc_id):
        ...     sources = {"doc1": "Abel soñaba ser músico."}
        ...     return sources.get(doc_id, "")
        >>> summary_by_item = {
        ...     "vic_grupo_social": [{"span": "músico", "doc_id": "doc1"}]
        ... }
        >>> result = compute_span_offsets(summary_by_item, get_source)
        >>> result["vic_grupo_social"][0]["offset"]
        16
    """
