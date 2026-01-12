import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from json_repair import repair_json


@dataclass
class ProcessorResult:
    """
    Generic result container for a single LLM-backed processor call.

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
    task_name: str
    model_name: str

    declared_fields: Dict[str, Any]
    values: Dict[str, Any]

    input_text: str
    input_struct: Optional[Dict[str, Any]]

    output_text: str
    output_struct: Optional[Dict[str, Any]]

    doc_id: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_llm_call(
        cls,
        *,
        task_name: str,
        model_name: str,
        request_kwargs: Dict[str, Any],
        response: Any,
        doc_id: Optional[Any] = None,
    ) -> "ProcessorResult":
        """
        Build a ProcessorResult from the request kwargs and LLM response.

        This method is robust to partial failures: if parsing of either the
        input or output JSON fails, an error string is recorded while keeping
        the raw text available for debugging.
        """
        error_messages: List[str] = []

        # Extract input text (first user message content) and try to parse it.
        input_text = ""
        input_struct: Optional[Dict[str, Any]] = None
        try:
            messages = request_kwargs.get("messages") or []
            if messages:
                input_text = str(messages[0].get("content", ""))
                try:
                    repaired = repair_json(input_text)
                    input_struct = json.loads(repaired)
                except Exception as exc:  # noqa: BLE001
                    error_messages.append(f"Failed to parse input_struct: {exc}")
        except Exception as exc:  # noqa: BLE001
            error_messages.append(f"Failed to extract input_text: {exc}")

        # Extract raw output text.
        output_text = ""
        output_struct: Optional[Dict[str, Any]] = None
        try:
            output_text = response.choices[0].message.content
            try:
                repaired_out = repair_json(output_text)
                output_struct = json.loads(repaired_out)
            except Exception as exc:  # noqa: BLE001
                error_messages.append(f"Failed to parse output_struct: {exc}")
        except Exception as exc:  # noqa: BLE001
            error_messages.append(f"Failed to extract output_text: {exc}")

        # Determine declared fields from guided_json schema if available.
        declared_fields: Dict[str, Any] = {}
        try:
            guided = (
                request_kwargs.get("extra_body", {})
                .get("guided_json", {})
                .get("properties", {})
            )
            if isinstance(guided, dict):
                declared_fields = guided
        except Exception as exc:  # noqa: BLE001
            error_messages.append(f"Failed to read guided_json properties: {exc}")

        # Extract values for the declared fields from the parsed output.
        values: Dict[str, Any] = {}
        if output_struct is not None:
            for field_name in declared_fields.keys():
                values[field_name] = output_struct.get(field_name)

        error = "; ".join(error_messages) if error_messages else None

        return cls(
            task_name=task_name,
            model_name=model_name,
            declared_fields=declared_fields,
            values=values,
            input_text=input_text,
            input_struct=input_struct,
            output_text=output_text,
            output_struct=output_struct,
            doc_id=doc_id,
            error=error,
            metadata={},
        )

    def has_field(self, name: str) -> bool:
        """Return True if this result declares the given field."""
        return name in self.declared_fields

    def get(self, name: str, default: Any = None) -> Any:
        """Convenience accessor for a field value."""
        return self.values.get(name, default)

    def is_no_info(self) -> bool:
        """
        Heuristic check for a "no information" style result.

        This intentionally stays generic and relies on commonly used fields:
        - For summary-like tasks, an empty or missing 'summary' suggests
          no information.
        - For classification-like tasks, an empty or missing 'result'
          suggests no information.
        """
        if "summary" in self.values:
            summary = (self.values.get("summary") or "").strip()
            return not summary
        if "result" in self.values:
            result = (self.values.get("result") or "").strip()
            return not result
        return False


class MessyTextLogicMixin:
    """
    Contains pure logic for MessyTextProcessor.
    Handles text cleaning, prompt construction, and response parsing.
    Decoupled from I/O (client calls).
    """
    def __init__(self, config: Dict[str, Any], taxonomy: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the logic mixin with configuration and taxonomy.

        Args:
            config (Dict[str, Any]): Runtime settings (model name, tokens, etc).
            taxonomy (Dict[str, Any]): Static definitions (context_definitions, label_options).
            logger (logging.Logger): Logger instance.
        """
        self.config = config
        self.definitions = taxonomy['context_definitions']
        self.labels = taxonomy['label_options']
        self.logger = logger
        self.model_name = config['model']['name']

        # Storage for the most recent per-call results.
        # These are additive: they do not replace any existing state or behavior.
        self.last_summary_result: Optional[ProcessorResult] = None
        self.last_classification_result: Optional[ProcessorResult] = None

    def clean_text(self, text: str) -> str:
        """
        Applies regex cleaning to input text.

        Args:
            text (str): The raw input text to clean.

        Returns:
            str: The cleaned text with URLs removed and whitespace normalized.
        """
        if not isinstance(text, str):
            return str(text)
            
        # Regex from opr_3.1
        # 1. Remove URLs, parenthetical metadata, emojis, date patterns like d/d
        text = re.sub(r'https?://\S+|\([^)]*/[^)]*\)|[\ue000-\uf8ff]|\b\d/\d\b', '', text)
        # 2. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_summary_args(self, text: str) -> Dict[str, Any]:
        """
        Constructs arguments for the summary API call.

        Args:
            text (str): The cleaned text to summarize.

        Returns:
            Dict[str, Any]: A dictionary containing model arguments (model, messages, etc.).
        """
        prompts_cfg = self.config.get("prompts") or {}
        summary_cfg = prompts_cfg.get("summary") or {}
        output_format = summary_cfg.get("output_format")
        instructions_template = summary_cfg.get("instructions")

        if output_format is None or instructions_template is None:
            raise ValueError(
                "Summary prompt configuration must provide 'output_format' and 'instructions'."
            )

        instructions = [
            instr.format(
                info_found_label="info_found",
                relevant_context_label="relevant_context",
                summary_label="summary",
                input_text_label="input_text",
            )
            for instr in instructions_template
        ]

        prompt_structure = {
            'input_text': text,
            'related_context': self.definitions,
            'output_format': output_format,
            'instructions': instructions,
        }
        
        # We cast to string to mimic notebook behavior of passing stringified dict
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
                    "summary": {"type": "string"}
                }, 
                "required": ["info_found", "relevant_context", "summary"]
            }}
        }

    def _get_conversation_summary_args(
        self,
        previous_summary: Optional[str],
        text: str,
    ) -> Dict[str, Any]:
        """
        Constructs arguments for a conversation-style summary API call.

        This method selects between first-turn and update prompts based on the
        previous_summary value and exposes both the previous summary and the new
        document to the model.

        Args:
            previous_summary (Optional[str]): The running summary from earlier
                turns, or None/empty string for the first turn.
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

        # Derive previous_relevant_context and previous_summary_by_item from the last
        # structured conversation result, if available. This allows prompt templates
        # to reference {previous_relevant_context} and {previous_summary_by_item}
        # without changing the public interface.
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

        # Allow prompt templates to reference previous_summary, previous structured
        # fields, and label names dynamically.
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
                    "summary_by_item": {"type": "object"},
                    "summary": {"type": "string"}
                },
                "required": ["info_found", "relevant_context", "summary"]
            }}
        }

    def _extract_summary_from_response(self, response: Any) -> str:
        """
        Parses the summary API response.

        Args:
            response (Any): The raw response object from the API call.

        Returns:
            str: The extracted summary text, or 'No relevant information found'.
        """
        content = response.choices[0].message.content
        parsed = json.loads(repair_json(content))
        summary = parsed.get('summary', 'No relevant information found')
        return summary if summary else 'No relevant information found'

    def _get_classification_args(self, summary: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Constructs arguments for the classification API call.

        Args:
            summary (str): The summarized text to classify.
            key (str): The taxonomy key (e.g. 'vic_grupo_social').

        Returns:
            Optional[Dict[str, Any]]: A dictionary of arguments, or None if validation fails.
        """
        if not summary or summary == 'No relevant information found':
            return None
        
        question = self.definitions.get(key, '')
        possible_values = self.labels.get(key, [])
        
        if not question:
            self.logger.error(f"Key '{key}' not found in definitions")
            return None

        prompts_cfg = self.config.get("prompts") or {}
        classification_cfg = prompts_cfg.get("classification") or {}
        instructions_template = classification_cfg.get("instructions")

        if not instructions_template:
            raise ValueError(
                "Classification prompt configuration must provide 'instructions' list."
            )

        # Apply formatting uniformly so any placeholder like {possible_values}
        # can be replaced, while literal JSON braces are preserved via {{ }} in
        # the template strings.
        instructions = [
            instr.format(possible_values=possible_values)
            for instr in instructions_template
        ]

        prompt_structure = {
            'input_text': summary,
            'question': question,
            'possible_values': possible_values,
            'instructions': instructions,
        }
        
        prompt_content = str(prompt_structure)

        return {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': prompt_content}],
            "temperature": self.config['processing']['temperature'],
            "max_tokens": self.config['processing']['max_tokens_classification'],
            "extra_body": {"guided_json": {
                "type": "object", 
                "properties": {
                    "evidence": {"type": "string"}, 
                    "result": {"type": "string"}
                }, 
                "required": ["evidence", "result"]
            }}
        }

    def _extract_classification_from_response(self, response: Any) -> str:
        """
        Parses the classification API response.

        Args:
            response (Any): The raw response object from the API call.

        Returns:
            str: The extracted classification result, or 'No information'.
        """
        content = response.choices[0].message.content
        parsed = json.loads(repair_json(content))
        result = parsed.get('result', 'No information')
        return result if result else 'No information'


class MessyTextProcessor(MessyTextLogicMixin):
    """
    Synchronous processor implementation.
    Maintains exact backward compatibility with original implementation.
    """
    def __init__(self, client: Any, config: Dict[str, Any], taxonomy: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the synchronous processor.

        Args:
            client (OpenAI): The synchronous OpenAI/vLLM client instance.
            config (Dict[str, Any]): Runtime settings (model name, tokens, etc).
            taxonomy (Dict[str, Any]): Static definitions (context_definitions, label_options).
            logger (logging.Logger): Logger instance.
        """
        super().__init__(config, taxonomy, logger)
        self.client = client

    def summarize_text(self, text: str, doc_id: Optional[Any] = None) -> str:
        """
        Run a summarization call, store a ProcessorResult, and return the summary text.

        This is the single public API for non-conversational summarization.
        Existing callers that expect a string remain compatible; richer
        consumers can read self.last_summary_result.
        """
        # Empty input: record a trivial "no information" result and return sentinel.
        if not text:
            self.last_summary_result = ProcessorResult(
                task_name="summary",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            return "No relevant information found"

        kwargs = self._get_summary_args(text)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Summarization failed: {e}")
            # Store an error result so callers can still inspect input.
            messages = kwargs.get("messages", []) or [{}]
            input_text = str(messages[0].get("content", ""))
            self.last_summary_result = ProcessorResult(
                task_name="summary",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text=input_text,
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=str(e),
                metadata={},
            )
            return "No relevant information found"

        # Normal path: construct and store the full result, then return summary text.
        self.last_summary_result = ProcessorResult.from_llm_call(
            task_name="summary",
            model_name=self.model_name,
            request_kwargs=kwargs,
            response=response,
            doc_id=doc_id,
        )
        summary = self.last_summary_result.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        return "No relevant information found"

    def classify_summary(
        self,
        summary: str,
        key: str,
        doc_id: Optional[Any] = None,
    ) -> str:
        """
        Run a classification call, store a ProcessorResult, and return the result string.

        This is the single public API for non-conversational classification.
        """
        kwargs = self._get_classification_args(summary, key)
        if kwargs is None:
            # Nothing to classify (e.g. no summary): record an empty result.
            self.last_classification_result = ProcessorResult(
                task_name="classification",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            return "No information"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Classification failed for {key}: {e}")
            messages = kwargs.get("messages", []) or [{}]
            input_text = str(messages[0].get("content", ""))
            self.last_classification_result = ProcessorResult(
                task_name="classification",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text=input_text,
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=str(e),
                metadata={},
            )
            return "No information"

        self.last_classification_result = ProcessorResult.from_llm_call(
            task_name="classification",
            model_name=self.model_name,
            request_kwargs=kwargs,
            response=response,
            doc_id=doc_id,
        )
        classification = self.last_classification_result.get("result")
        if isinstance(classification, str) and classification.strip():
            return classification
        return "No information"


class AsyncMessyTextProcessor(MessyTextLogicMixin):
    """
    Asynchronous processor implementation for high-throughput servers.
    Uses async/await patterns with AsyncOpenAI client.
    """
    def __init__(self, client: Any, config: Dict[str, Any], taxonomy: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the asynchronous processor.

        Args:
            client (AsyncOpenAI): The asynchronous OpenAI/vLLM client instance.
            config (Dict[str, Any]): Runtime settings (model name, tokens, etc).
            taxonomy (Dict[str, Any]): Static definitions (context_definitions, label_options).
            logger (logging.Logger): Logger instance.
        """
        super().__init__(config, taxonomy, logger)
        self.client = client

    async def summarize_text(self, text: str, doc_id: Optional[Any] = None) -> str:
        """
        Async summarization API: store a ProcessorResult and return the summary text.
        """
        if not text:
            self.last_summary_result = ProcessorResult(
                task_name="summary",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            return "No relevant information found"

        kwargs = self._get_summary_args(text)
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Summarization failed: {e}")
            messages = kwargs.get("messages", []) or [{}]
            input_text = str(messages[0].get("content", ""))
            self.last_summary_result = ProcessorResult(
                task_name="summary",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text=input_text,
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=str(e),
                metadata={},
            )
            return "No relevant information found"

        self.last_summary_result = ProcessorResult.from_llm_call(
            task_name="summary",
            model_name=self.model_name,
            request_kwargs=kwargs,
            response=response,
            doc_id=doc_id,
        )
        summary = self.last_summary_result.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary
        return "No relevant information found"

    async def classify_summary(
        self,
        summary: str,
        key: str,
        doc_id: Optional[Any] = None,
    ) -> str:
        """
        Async classification API: store a ProcessorResult and return the result string.
        """
        kwargs = self._get_classification_args(summary, key)
        if kwargs is None:
            self.last_classification_result = ProcessorResult(
                task_name="classification",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            return "No information"

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Classification failed for {key}: {e}")
            messages = kwargs.get("messages", []) or [{}]
            input_text = str(messages[0].get("content", ""))
            self.last_classification_result = ProcessorResult(
                task_name="classification",
                model_name=self.model_name,
                declared_fields={},
                values={},
                input_text=input_text,
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=str(e),
                metadata={},
            )
            return "No information"

        self.last_classification_result = ProcessorResult.from_llm_call(
            task_name="classification",
            model_name=self.model_name,
            request_kwargs=kwargs,
            response=response,
            doc_id=doc_id,
        )
        classification = self.last_classification_result.get("result")
        if isinstance(classification, str) and classification.strip():
            return classification
        return "No information"


@dataclass
class MessyTextConversationState:
    """
    Holds lightweight state for a multi-turn MessyText conversation.

    This object is intentionally minimal and focused on summarization so it can be
    extended later without breaking the interface. For now, it keeps track of the
    most recent summary, the number of processed turns, and the most recent
    structured result object.

    Attributes:
        last_summary (str): Summary produced in the most recent turn.
        turn_index (int): Number of turns that have been processed so far.
        last_result (Optional[ProcessorResult]): Structured result for the most
            recent turn, including fields like 'info_found' and 'summary'.
    """
    last_summary: str = ""
    turn_index: int = 0
    last_result: Optional[ProcessorResult] = None


class MessyTextConversationTurnProcessor:
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

    def __init__(self, processor: MessyTextProcessor) -> None:
        """
        Initializes the turn processor.

        Args:
            processor (MessyTextProcessor): The underlying synchronous processor
                responsible for cleaning, prompt construction, and LLM calls.
        """
        self.processor = processor

    def process_turn(
        self,
        raw_text: str,
        state: Optional[MessyTextConversationState] = None,
        doc_id: Optional[Any] = None,
    ) -> Tuple[str, MessyTextConversationState]:
        """
        Processes a single document as one conversation turn.

        Args:
            raw_text (str): The raw document text for this turn.
            state (Optional[MessyTextConversationState]): Existing conversation
                state to carry over between turns. If None, a new state is created.

        Returns:
            Tuple[str, MessyTextConversationState]:
                - The summary generated for this turn.
                - The updated conversation state including the latest summary and
                  incremented turn index, plus the structured ProcessorResult for
                  this turn in state.last_result.
        """
        conversation_state = state or MessyTextConversationState()

        cleaned_text = self.processor.clean_text(raw_text)
        result: Optional[ProcessorResult] = None

        if not cleaned_text.strip():
            # No input text: record an explicit "no information" result and use
            # an empty summary string so callers can rely on info_found/summary
            # instead of sentinel phrases.
            result = ProcessorResult(
                task_name="conversation_summary",
                model_name=self.processor.model_name,
                declared_fields={},
                values={"info_found": "FALSE", "summary": "", "relevant_context": []},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            summary = ""
        else:
            kwargs = self.processor._get_conversation_summary_args(
                previous_summary=conversation_state.last_summary,
                text=cleaned_text,
            )
            try:
                response = self.processor.client.chat.completions.create(**kwargs)
                # Construct a structured result from the guided JSON output.
                result = ProcessorResult.from_llm_call(
                    task_name="conversation_summary",
                    model_name=self.processor.model_name,
                    request_kwargs=kwargs,
                    response=response,
                    doc_id=doc_id,
                )
                summary_value = result.get("summary")
                if isinstance(summary_value, str) and summary_value.strip():
                    summary = summary_value
                else:
                    summary = ""
            except Exception as e:
                self.processor.logger.error(f"Summarization (conversation) failed: {e}")
                result = ProcessorResult(
                    task_name="conversation_summary",
                    model_name=self.processor.model_name,
                    declared_fields={},
                    values={"info_found": "FALSE", "summary": "", "relevant_context": []},
                    input_text="",
                    input_struct=None,
                    output_text="",
                    output_struct=None,
                    doc_id=doc_id,
                    error=str(e),
                    metadata={},
                )
                summary = ""

        # Expose the most recent conversation result on the underlying processor
        # for debugging or richer consumers.
        self.processor.last_summary_result = result

        # Optional: per-turn logging when log_response is enabled
        log_cfg = (getattr(self.processor, "config", {}) or {}).get("logging", {}) or {}
        if log_cfg.get("log_response") and result is not None:
            # 1) one line: input prompt
            self.processor.logger.info(
                "conversation_input sync turn=%d doc_id=%r has_summary=%s INPUT=%r",
                conversation_state.turn_index,
                result.doc_id,
                bool(summary),
                result.input_text,
            )
            # 2) one line: raw model output
            self.processor.logger.info(
                "conversation_output sync turn=%d doc_id=%r OUTPUT=%r",
                conversation_state.turn_index,
                result.doc_id,
                result.output_text,
            )
            # 3) one line: full ProcessorResult object
            self.processor.logger.info(
                "conversation_result sync turn=%d doc_id=%r result=%r",
                conversation_state.turn_index,
                result.doc_id,
                result,
            )

        updated_state = MessyTextConversationState(
            last_summary=summary,
            turn_index=conversation_state.turn_index + 1,
            last_result=result,
        )
        return summary, updated_state


class MessyTextConversationOrchestrator:
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

    def __init__(self, turn_processor: MessyTextConversationTurnProcessor) -> None:
        """
        Initializes the orchestrator.

        Args:
            turn_processor (MessyTextConversationTurnProcessor): The per-turn
                processor used to handle individual documents.
        """
        self.turn_processor = turn_processor

    def run_conversation(
        self,
        texts: Iterable[str],
        initial_state: Optional[MessyTextConversationState] = None,
        stop_condition: Optional[
            Callable[[MessyTextConversationState, str], bool]
        ] = None,
    ) -> Tuple[List[str], MessyTextConversationState]:
        """
        Runs a sequential conversation over a collection of texts.

        Args:
            texts (Iterable[str]): Iterable of raw document texts to process
                sequentially (one per turn).
            initial_state (Optional[MessyTextConversationState]): Starting
                conversation state. If None, a new empty state is created.
            stop_condition (Optional[Callable[[MessyTextConversationState, str], bool]]):
                Optional callback that receives the current state and the most
                recent summary. If it returns True, the conversation stops after
                that turn.

        Returns:
            Tuple[List[str], MessyTextConversationState]:
                - List of per-turn summaries in the order they were processed.
                - Final conversation state after the last processed turn.
        """
        state = initial_state or MessyTextConversationState()
        summaries: List[str] = []

        for raw_text in texts:
            summary, state = self.turn_processor.process_turn(raw_text, state)
            summaries.append(summary)

            if stop_condition is not None and stop_condition(state, summary):
                break

        return summaries, state


class AsyncMessyTextConversationTurnProcessor:
    """
    Asynchronous counterpart of MessyTextConversationTurnProcessor.

    This class wraps an AsyncMessyTextProcessor and exposes an async interface
    for running a single summarization turn with optional conversation state.
    """

    def __init__(self, processor: AsyncMessyTextProcessor) -> None:
        """
        Initializes the async turn processor.

        Args:
            processor (AsyncMessyTextProcessor): The underlying asynchronous
                processor responsible for cleaning, prompt construction, and
                LLM calls.
        """
        self.processor = processor

    async def process_turn(
        self,
        raw_text: str,
        state: Optional[MessyTextConversationState] = None,
        doc_id: Optional[Any] = None,
    ) -> Tuple[str, MessyTextConversationState]:
        """
        Asynchronously processes a single document as one conversation turn.

        Args:
            raw_text (str): The raw document text for this turn.
            state (Optional[MessyTextConversationState]): Existing conversation
                state to carry over between turns. If None, a new state is created.

        Returns:
            Tuple[str, MessyTextConversationState]:
                - The summary generated for this turn.
                - The updated conversation state including the latest summary and
                  incremented turn index, plus the structured ProcessorResult for
                  this turn in state.last_result.
        """
        conversation_state = state or MessyTextConversationState()

        cleaned_text = self.processor.clean_text(raw_text)
        result: Optional[ProcessorResult] = None

        if not cleaned_text.strip():
            result = ProcessorResult(
                task_name="conversation_summary",
                model_name=self.processor.model_name,
                declared_fields={},
                values={"info_found": "FALSE", "summary": "", "relevant_context": []},
                input_text="",
                input_struct=None,
                output_text="",
                output_struct=None,
                doc_id=doc_id,
                error=None,
                metadata={},
            )
            summary = ""
        else:
            kwargs = self.processor._get_conversation_summary_args(
                previous_summary=conversation_state.last_summary,
                text=cleaned_text,
            )
            try:
                response = await self.processor.client.chat.completions.create(**kwargs)
                result = ProcessorResult.from_llm_call(
                    task_name="conversation_summary",
                    model_name=self.processor.model_name,
                    request_kwargs=kwargs,
                    response=response,
                    doc_id=doc_id,
                )
                summary_value = result.get("summary")
                if isinstance(summary_value, str) and summary_value.strip():
                    summary = summary_value
                else:
                    summary = ""
            except Exception as e:
                self.processor.logger.error(f"Summarization (conversation) failed: {e}")
                result = ProcessorResult(
                    task_name="conversation_summary",
                    model_name=self.processor.model_name,
                    declared_fields={},
                    values={"info_found": "FALSE", "summary": "", "relevant_context": []},
                    input_text="",
                    input_struct=None,
                    output_text="",
                    output_struct=None,
                    doc_id=doc_id,
                    error=str(e),
                    metadata={},
                )
                summary = ""

        self.processor.last_summary_result = result

        log_cfg = (getattr(self.processor, "config", {}) or {}).get("logging", {}) or {}
        if log_cfg.get("log_response") and result is not None:
            # 1) one line: input prompt
            self.processor.logger.info(
                "conversation_input async turn=%d doc_id=%r has_summary=%s INPUT=%r",
                conversation_state.turn_index,
                result.doc_id,
                bool(summary),
                result.input_text,
            )
            # 2) one line: raw model output
            self.processor.logger.info(
                "conversation_output async turn=%d doc_id=%r OUTPUT=%r",
                conversation_state.turn_index,
                result.doc_id,
                result.output_text,
            )
            # 3) one line: full ProcessorResult object
            self.processor.logger.info(
                "conversation_result async turn=%d doc_id=%r result=%r",
                conversation_state.turn_index,
                result.doc_id,
                result,
            )

        updated_state = MessyTextConversationState(
            last_summary=summary,
            turn_index=conversation_state.turn_index + 1,
            last_result=result,
        )
        return summary, updated_state


class AsyncMessyTextConversationOrchestrator:
    """
    Asynchronous orchestrator for multi-turn MessyText conversations.

    This class mirrors MessyTextConversationOrchestrator but exposes an async
    run_conversation method so that multiple conversations can be executed
    concurrently by higher-level pipelines.
    """

    def __init__(self, turn_processor: AsyncMessyTextConversationTurnProcessor) -> None:
        """
        Initializes the async orchestrator.

        Args:
            turn_processor (AsyncMessyTextConversationTurnProcessor): The per-turn
                processor used to handle individual documents asynchronously.
        """
        self.turn_processor = turn_processor

    async def run_conversation(
        self,
        texts: Iterable[str],
        initial_state: Optional[MessyTextConversationState] = None,
        stop_condition: Optional[
            Callable[[MessyTextConversationState, str], bool]
        ] = None,
    ) -> Tuple[List[str], MessyTextConversationState]:
        """
        Asynchronously runs a sequential conversation over a collection of texts.

        Args:
            texts (Iterable[str]): Iterable of raw document texts to process
                sequentially (one per turn).
            initial_state (Optional[MessyTextConversationState]): Starting
                conversation state. If None, a new empty state is created.
            stop_condition (Optional[Callable[[MessyTextConversationState, str], bool]]):
                Optional callback that receives the current state and the most
                recent summary. If it returns True, the conversation stops after
                that turn.

        Returns:
            Tuple[List[str], MessyTextConversationState]:
                - List of per-turn summaries in the order they were processed.
                - Final conversation state after the last processed turn.
        """
        state = initial_state or MessyTextConversationState()
        summaries: List[str] = []

        for raw_text in texts:
            summary, state = await self.turn_processor.process_turn(raw_text, state)
            summaries.append(summary)

            if stop_condition is not None and stop_condition(state, summary):
                break

        return summaries, state
