"""
Why not using comet or bleu to evaluate the model? They are for translation tasks.
Why not using rouge or similar benchmarks to evaluate the model? They requires human reference.
QAFactEval or G-Eval works, but they are QA based. 
We also use regular benchmarks like accuracy, precision, recall, f1-score, etc.

adapted QAFactEval:
They use: QA from sources vs QA from summaries, and get LERC scores.
We use: Machine labelling from sources vs machine labelling from summaries, and get LERC scores.

G-Eval:
Uses DeepEval's built-in SummarizationMetric and HallucinationMetric.
Reference: https://github.com/confident-ai/deepeval
"""

import os
import logging
from typing import Dict, Any

from deepeval.metrics import SummarizationMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase


class GEvalEvaluator:
    """
    Evaluates summary quality using DeepEval's built-in metrics.
    Provides methods for summarization and hallucination evaluation.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the G-Eval evaluator with API configuration.

        Args:
            config (Dict[str, Any]): Runtime settings containing 'geval.api_key' for OpenAI API.
            logger (logging.Logger): Logger instance for error/info messages.
        """
        self.config = config
        self.logger = logger
        
        # Set API key from config if provided
        api_key = config.get("geval", {}).get("api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    def evaluate_summarization(self, source: str, summary: str) -> float:
        """
        Evaluates summary quality using DeepEval's SummarizationMetric.
        Measures alignment (no hallucination) and inclusion (key points covered).

        Args:
            source (str): The original source document text.
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1 range).
        """
        metric = SummarizationMetric(threshold=0.5)
        
        test_case = LLMTestCase(
            input=source,
            actual_output=summary
        )

        try:
            metric.measure(test_case)
            score = metric.score
            self.logger.info(f"Summarization score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Summarization evaluation failed: {e}")
            return 0.0

    def evaluate_hallucination(self, source: str, summary: str) -> float:
        """
        Evaluates hallucination using DeepEval's HallucinationMetric.
        Checks if summary contains statements not supported by source.

        Args:
            source (str): The original source document text (used as context).
            summary (str): The machine-generated summary text.

        Returns:
            float: The evaluation score (0-1, higher = less hallucination).
        """
        metric = HallucinationMetric(threshold=0.5)
        
        test_case = LLMTestCase(
            input=source,
            actual_output=summary,
            context=[source]  # Source text as ground truth context
        )

        try:
            metric.measure(test_case)
            score = metric.score
            self.logger.info(f"Hallucination score: {score:.3f}")
            return score
        except Exception as e:
            self.logger.error(f"Hallucination evaluation failed: {e}")
            return 0.0
