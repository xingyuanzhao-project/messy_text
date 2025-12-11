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

import json
import logging
from typing import Dict, Any, Optional, Union

from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import SummarizationMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase


# =============================================================================
# VLLMModel: Logic Mixin and Sync/Async Implementations
# =============================================================================

class VLLMModelLogicMixin:
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
    def __init__(self, model_name: str, temperature: float = 0):
        """
        Initializes the logic mixin with model configuration.

        Args:
            model_name (str): The full model name (e.g., 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4').
            temperature (float): Temperature for generation. Defaults to 0.
        """
        self._model_name = model_name
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_model_name(self) -> str:
        """Returns the model name for display."""
        return f"{self._model_name} (vLLM)"

    def _parse_response(self, content: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Parses the response content, optionally validating against schema.

        Args:
            content (str): Raw response content from the model.
            schema (Optional[BaseModel]): Pydantic schema for structured output.

        Returns:
            Union[str, BaseModel]: Parsed content or validated schema instance.
        """
        if schema:
            json_output = json.loads(content)
            return schema.model_validate(json_output)
        return content


class VLLMModel(VLLMModelLogicMixin, DeepEvalBaseLLM):
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
    def __init__(self, client: Any, model_name: str, temperature: float = 0):
        """
        Initializes the synchronous vLLM model wrapper.

        Args:
            client (OpenAI): The synchronous OpenAI-compatible client instance.
            model_name (str): The full model name.
            temperature (float): Temperature for generation. Defaults to 0.
        """
        VLLMModelLogicMixin.__init__(self, model_name, temperature)
        self.client = client

    def load_model(self) -> Any:
        """Returns the OpenAI client."""
        return self.client

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Generates a response from the model (sync).

        Args:
            prompt (str): The prompt to send to the model.
            schema (Optional[BaseModel]): Pydantic schema for structured output.

        Returns:
            Union[str, BaseModel]: The response content (str or validated schema).
        """
        response = self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        content = response.choices[0].message.content
        return self._parse_response(content, schema)

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Async generation (falls back to sync for this class).

        Args:
            prompt (str): The prompt to send to the model.
            schema (Optional[BaseModel]): Pydantic schema for structured output.

        Returns:
            Union[str, BaseModel]: The response content (str or validated schema).
        """
        return self.generate(prompt, schema)


class AsyncVLLMModel(VLLMModelLogicMixin, DeepEvalBaseLLM):
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
    def __init__(self, client: Any, model_name: str, temperature: float = 0):
        """
        Initializes the asynchronous vLLM model wrapper.

        Args:
            client (AsyncOpenAI): The asynchronous OpenAI-compatible client instance.
            model_name (str): The full model name.
            temperature (float): Temperature for generation. Defaults to 0.
        """
        VLLMModelLogicMixin.__init__(self, model_name, temperature)
        self.client = client

    def load_model(self) -> Any:
        """Returns the AsyncOpenAI client."""
        return self.client

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Sync generation (not supported for async model, raises error).

        Args:
            prompt (str): The prompt to send to the model.
            schema (Optional[BaseModel]): Pydantic schema for structured output.

        Raises:
            RuntimeError: AsyncVLLMModel does not support sync generate.
        """
        raise RuntimeError("AsyncVLLMModel does not support sync generate(). Use a_generate() instead.")

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """
        Generates a response from the model (async).

        Args:
            prompt (str): The prompt to send to the model.
            schema (Optional[BaseModel]): Pydantic schema for structured output.

        Returns:
            Union[str, BaseModel]: The response content (str or validated schema).
        """
        response = await self.client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        content = response.choices[0].message.content
        return self._parse_response(content, schema)


# =============================================================================
# GEvalEvaluator: Logic Mixin and Sync/Async Implementations
# =============================================================================

class GEvalEvaluatorLogicMixin:
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
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the logic mixin with configuration.

        Args:
            config (Dict[str, Any]): Runtime settings.
            logger (logging.Logger): Logger instance for error/info messages.
        """
        self.config = config
        self.logger = logger

    def _create_summarization_metric(self, model: DeepEvalBaseLLM) -> SummarizationMetric:
        """
        Creates a SummarizationMetric instance.

        Args:
            model (DeepEvalBaseLLM): The model wrapper to use for evaluation.

        Returns:
            SummarizationMetric: Configured metric instance.
        """
        return SummarizationMetric(threshold=0.5, model=model)

    def _create_hallucination_metric(self, model: DeepEvalBaseLLM) -> HallucinationMetric:
        """
        Creates a HallucinationMetric instance.

        Args:
            model (DeepEvalBaseLLM): The model wrapper to use for evaluation.

        Returns:
            HallucinationMetric: Configured metric instance.
        """
        return HallucinationMetric(threshold=0.5, model=model)

    def _create_summarization_test_case(self, source: str, summary: str) -> LLMTestCase:
        """
        Creates a test case for summarization evaluation.

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            LLMTestCase: Configured test case instance.
        """
        return LLMTestCase(input=source, actual_output=summary)

    def _create_hallucination_test_case(self, source: str, summary: str) -> LLMTestCase:
        """
        Creates a test case for hallucination evaluation.

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            LLMTestCase: Configured test case instance with context.
        """
        return LLMTestCase(input=source, actual_output=summary, context=[source])


class GEvalEvaluator(GEvalEvaluatorLogicMixin):
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
    def __init__(self, client: Any, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the synchronous G-Eval evaluator.

        Args:
            client (OpenAI): The synchronous OpenAI-compatible client instance.
            config (Dict[str, Any]): Runtime settings containing 'model.name'.
            logger (logging.Logger): Logger instance for error/info messages.
        """
        super().__init__(config, logger)
        self.model = VLLMModel(
            client=client,
            model_name=config['model']['name'],
            temperature=0
        )
        self.logger.info(f"GEvalEvaluator initialized with model: {self.model.get_model_name()}")

    def evaluate_summarization(self, source: str, summary: str) -> float:
        """
        Evaluates summary quality using DeepEval's SummarizationMetric (sync).

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1 range).
        """
        metric = self._create_summarization_metric(self.model)
        test_case = self._create_summarization_test_case(source, summary)

        try:
            metric.measure(test_case, _show_indicator=False)
            score = metric.score
            self.logger.info(f"Summarization score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Summarization evaluation failed: {e}")
            return 0.0

    def evaluate_hallucination(self, source: str, summary: str) -> float:
        """
        Evaluates hallucination using DeepEval's HallucinationMetric (sync).

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1, higher = less hallucination).
        """
        metric = self._create_hallucination_metric(self.model)
        test_case = self._create_hallucination_test_case(source, summary)

        try:
            metric.measure(test_case, _show_indicator=False)
            score = metric.score
            self.logger.info(f"Hallucination score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Hallucination evaluation failed: {e}")
            return 0.0


class AsyncGEvalEvaluator(GEvalEvaluatorLogicMixin):
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
    def __init__(self, client: Any, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the asynchronous G-Eval evaluator.

        Args:
            client (AsyncOpenAI): The asynchronous OpenAI-compatible client instance.
            config (Dict[str, Any]): Runtime settings containing 'model.name'.
            logger (logging.Logger): Logger instance for error/info messages.
        """
        super().__init__(config, logger)
        self.model = AsyncVLLMModel(
            client=client,
            model_name=config['model']['name'],
            temperature=0
        )
        self.logger.info(f"AsyncGEvalEvaluator initialized with model: {self.model.get_model_name()}")

    async def evaluate_summarization(self, source: str, summary: str) -> float:
        """
        Evaluates summary quality using DeepEval's SummarizationMetric (async).

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1 range).
        """
        metric = self._create_summarization_metric(self.model)
        test_case = self._create_summarization_test_case(source, summary)

        try:
            await metric.a_measure(test_case, _show_indicator=False)
            score = metric.score
            self.logger.info(f"Summarization score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Summarization evaluation failed: {e}")
            return 0.0

    async def evaluate_hallucination(self, source: str, summary: str) -> float:
        """
        Evaluates hallucination using DeepEval's HallucinationMetric (async).

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1, higher = less hallucination).
        """
        metric = self._create_hallucination_metric(self.model)
        test_case = self._create_hallucination_test_case(source, summary)

        try:
            await metric.a_measure(test_case, _show_indicator=False)
            score = metric.score
            self.logger.info(f"Hallucination score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Hallucination evaluation failed: {e}")
            return 0.0


# =============================================================================
# SummaCEvaluator: NLI-based Factual Consistency Evaluation
# =============================================================================

class SummaCEvaluator:
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
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the SummaC evaluator with both ZS and Conv models (eager loading).

        Args:
            config (Dict[str, Any]): Runtime settings containing 'evaluation.benchmarks.summac_settings' with:
                - device (str): "cuda" or "cpu"
                - granularity (str): "sentence" or "paragraph"
                - model_name (str): NLI model name (e.g., "vitc", "mnli")
            logger (logging.Logger): Logger instance for error/info messages.
        """
        from summac.model_summac import SummaCZS, SummaCConv

        self.config = config
        self.logger = logger

        # Extract SummaC configuration from evaluation.benchmarks.summac_settings
        summac_config = config.get('evaluation', {}).get('benchmarks', {}).get('summac_settings', {})
        device = summac_config.get('device', 'cuda')
        granularity = summac_config.get('granularity', 'sentence')
        model_name = summac_config.get('model_name', 'vitc')

        self.logger.info(f"Initializing SummaCEvaluator: device={device}, granularity={granularity}, model={model_name}")

        # Eager initialization: load both models at startup
        try:
            self.model_zs = SummaCZS(
                granularity=granularity,
                model_name=model_name,
                device=device
            )
            self.logger.info("SummaCZS model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SummaCZS: {e}")
            raise

        try:
            self.model_conv = SummaCConv(
                models=[model_name],
                bins='percentile',
                granularity=granularity,
                nli_labels='e',
                device=device,
                start_file=None,  # Skip pretrained weights, use lazy initialization
                agg='mean'
            )
            self.logger.info("SummaCConv model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load SummaCConv: {e}")
            raise

        self.logger.info("SummaCEvaluator initialized with both ZS and Conv models")

    def evaluate_zs(self, source: str, summary: str) -> float:
        """
        Evaluates factual consistency using SummaCZS (zero-shot NLI).

        SummaCZS uses direct NLI inference without learned aggregation.
        Faster but slightly less accurate than SummaCConv.

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: Consistency score (0-1, higher = more factually consistent).
        """
        if not source or not summary:
            self.logger.warning("Empty source or summary provided to evaluate_zs")
            return 0.0

        try:
            result = self.model_zs.score([source], [summary])
            score = result['scores'][0]
            self.logger.info(f"SummaCZS score: {score:.3f}")
            return float(score)
        except Exception as e:
            self.logger.error(f"SummaCZS evaluation failed: {e}")
            return 0.0

    def evaluate_conv(self, source: str, summary: str) -> float:
        """
        Evaluates factual consistency using SummaCConv (learned aggregation).

        SummaCConv uses convolutional layers for learned score aggregation.
        More accurate but slightly slower than SummaCZS.

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: Consistency score (0-1, higher = more factually consistent).
        """
        if not source or not summary:
            self.logger.warning("Empty source or summary provided to evaluate_conv")
            return 0.0

        try:
            result = self.model_conv.score([source], [summary])
            score = result['scores'][0]
            self.logger.info(f"SummaCConv score: {score:.3f}")
            return float(score)
        except Exception as e:
            self.logger.error(f"SummaCConv evaluation failed: {e}")
            return 0.0


# =============================================================================
# DefaultMetricsEvaluator: Classification Metrics (Accuracy, F1, Kappa)
# =============================================================================

class DefaultMetricsEvaluator:
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

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the default metrics evaluator and loads taxonomy.

        Args:
            config (Dict[str, Any]): Runtime settings containing:
                - paths.taxonomy (str): Path to taxonomy.json file.
            logger (logging.Logger): Logger instance for error/info messages.
        """
        self.config = config
        self.logger = logger

        # Load taxonomy to get field names (explicit config read, no fallback)
        taxonomy_path = config['paths']['taxonomy']

        try:
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                taxonomy = json.load(f)
            self.field_names = list(taxonomy.get('label_options', {}).keys())
            self.logger.info(f"DefaultMetricsEvaluator initialized with {len(self.field_names)} fields")
        except FileNotFoundError:
            self.logger.error(f"Taxonomy file not found: {taxonomy_path}")
            self.field_names = []
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse taxonomy JSON: {e}")
            self.field_names = []

    def evaluate_match(self, annotation: str, classification: str) -> float:
        """
        Evaluates per-row match between annotation and classification.

        Args:
            annotation (str): Ground truth label (human annotation).
            classification (str): Predicted label (machine classification).

        Returns:
            float: 1.0 if exact match, 0.0 otherwise.
        """
        y_true = str(annotation).strip() if annotation else ''
        y_pred = str(classification).strip() if classification else ''
        return 1.0 if y_true == y_pred else 0.0

    def get_column_pair(self, field_name: str) -> tuple:
        """
        Returns column name pair for a given field.

        Args:
            field_name (str): Field name from taxonomy (e.g., 'vic_grupo_social').

        Returns:
            tuple: (annotation_column, classification_column)
                   e.g., ('vic_grupo_social', 'vic_grupo_social_classification')
        """
        return (field_name, f"{field_name}_classification")
