## General case vs specific case

now the problems we encountered are not unique to this project

1. data source and or annotations do not match the unit of analysis that are required

2. then there is deduplication and extraction jobs that can be done with extractive ai

3. and we also show that we can carry out multiple tasks simultaneously with a stateful pipeline



8 19  6 7 8 12 19 appendix

14 15 combined, try bullet list them

leave out validation

3 ways of workflow showing: save them choose one


one audience for one ver



colleciton of tool, as major contribution



```json
  "summary_update": {
    "output_format": {
      "info_found": "<TRUE|FALSE>",
      "relevant_context": ["<label keys found>"],
      "summary_by_item": {
        "<label_key>": [
          {"span": "<exact SENTENCE from input text>"}
        ]
      },
      "summary": "<texto in spanish>"
    },
    "instructions": [
      "If the previous summary above is empty or only whitespace, base {info_found_label}, {relevant_context_label} and {summary_label} only on {input_text_label} (the new document).",
      "If the previous summary above is non-empty, update it by adding or refining victim-relevant facts from {input_text_label}, without removing correct existing details.",
      "If the input is an error/missing page (e.g., \"Página no encontrada\", \"404\", \"no se puede encontrar esa página\"), set {info_found_label}=\"FALSE\", {relevant_context_label}=[], summary_by_item={{}}, {summary_label}=\"\".",
      "Ignore navigation/site chrome (menú, buscar, categorías, compartir, ThemeGrill, WordPress, cookies, copyright).",
      "Use only the current {input_text_label}, {related_context_label} and {previous_summary} to decide the new {info_found_label}, {relevant_context_label} of your document",
      "if {previous_summary_by_item} is not empty, but you do not find any related information in {input_text_label}, keep the previous summary_by_item.",
      "For each key X in existing {previous_summary_by_item}, check if there is any new information about X in {input_text_label}. If yes, add the new span(s) to summary_by_item[X] as objects with 'span' (exact text). Keep existing spans from {previous_summary_by_item}.",
      "Each span must be a full original sentence ending in sentence punctuation (., ?, !, or equivalent) with no truncation or stitching; if you cannot find a full sentence for X, leave summary_by_item[X] as an empty list and do not invent placeholders.",
      "Extractive summary in Spanish: copy exact spans; DO NOT paraphrase; preserve modality (\"soñaba ser\", \"quería ser\", \"aspiraba a\").",
      "If no relevant info in either {previous_summary} or {input_text_label}, set {info_found_label}=\"FALSE\", {relevant_context_label}=[], summary_by_item={{}}, {summary_label}=\"\".",
      "NO APOLOGIES, NO FILLER TEXT"
    ]
  }
```

```python
def _get_conversation_summary_args(
        self,
        previous_summary: Optional[str],
        text: str,
    ) -> Dict[str, Any]:
        """
        Constructs arguments for a conversation-style multi-turn summary API call.

        This method selects between first-turn and update prompts based on if previous_summary value exist and passes both the previous summary and the new document to the model.

        Args:
            previous_summary (Optional[str]): The running summary from earlier turns, or None/empty string for the first turn.
            text (str): The cleaned text of the new document to integrate.

        Returns:
            Dict[str, Any]: A dictionary containing model arguments (model,
            messages, etc.) suitable for chat.completions.create.
        """
        prompts_cfg = self.config.get("prompts") or {}

        has_previous = bool(previous_summary and previous_summary.strip())
        if has_previous:
            summary_cfg = (
                prompts_cfg.get("summary_update")
                or prompts_cfg.get("summary_first")
                or prompts_cfg.get("summary")
                or {}
            )
        else:
            summary_cfg = (
                prompts_cfg.get("summary_first")
                or prompts_cfg.get("summary")
                or {}
            )

        previous_relevant_context_list = []
        previous_summary_by_item_dict = {}

        previous_result = getattr(self, "last_summary_result", None)
        if has_previous and isinstance(previous_result, ProcessorResult):
            raw_ctx = previous_result.get("relevant_context")
            if isinstance(raw_ctx, list):
                previous_relevant_context_list = raw_ctx

            raw_summary_by_item = previous_result.get("summary_by_item")
            if isinstance(raw_summary_by_item, dict):
                previous_summary_by_item_dict = raw_summary_by_item

        previous_relevant_context_str = json.dumps(
            previous_relevant_context_list, ensure_ascii=False
        )
        previous_summary_by_item_str = json.dumps(
            previous_summary_by_item_dict, ensure_ascii=False
        )

        output_format = summary_cfg.get("output_format")
        instructions_template = summary_cfg.get("instructions")

        if output_format is None or instructions_template is None:
            raise ValueError(
                "Conversation summary prompt configuration must provide "
                "'output_format' and 'instructions'."
            )

        # This allows prompt templates to reference previous_summary, previous structured fields, and label names dynamically.
        class _SafeFormatDict(dict):
            def __missing__(self, key: str) -> str:
                # Preserve unknown placeholders literally, e.g. {previous_relevant_context}
                return "{" + key + "}"

        format_vars = _SafeFormatDict(
            previous_summary=previous_summary or "",
            previous_relevant_context=previous_relevant_context_str,
            previous_summary_by_item=previous_summary_by_item_str,
            info_found_label="info_found",
            relevant_context_label="relevant_context",
            summary_label="summary",
            input_text_label="input_text",
            related_context_label="related_context",
        )

        instructions = [
            instr.format_map(format_vars)
            for instr in instructions_template
        ]

        prompt_structure = {
            'previous_summary': previous_summary or "",
            'input_text': text,
            'related_context': self.definitions,
            'output_format': output_format,
            'instructions': instructions,
        }

        prompt_content = str(prompt_structure)

        return {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': prompt_content}],
            "temperature": self.config['processing']['temperature'],
            "max_tokens": self.config['processing']['max_tokens_summary'],
            "extra_body": {"guided_json": {
                "type": "object",
                "properties": {
                    "info_found": {"type": "string"},
                    "relevant_context": {"type": "array"},
                    "summary_by_item": {
                        "type": "object",
                        "description": "Per-label extractive spans for traceback",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "span": {"type": "string"}
                                },
                                "required": ["span"]
                            }
                        }
                    },
                    "summary": {"type": "string"}
                },
                "required": ["info_found", "relevant_context", "summary_by_item", "summary"]
            }}
        }
```

```python

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
```