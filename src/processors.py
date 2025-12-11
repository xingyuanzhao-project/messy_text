import json
import re
import logging
from typing import Dict, Any, Optional

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
        prompt_structure = {
            'input_text': text,
            'related_context': self.definitions,
            'output_format': {
                'info_found': '<TRUE|FALSE>',
                'relevant_context': '<list of context keys found, or empty list>',
                'summary': '<texto in spanish>'
            },
            'instructions': [
                'If the input is an error/missing page (e.g., "Página no encontrada", "404", "no se puede encontrar esa página"), set info_found="FALSE", relevant_context=[], summary=""',
                'Ignore navigation/site chrome (menú, buscar, categorías, compartir, ThemeGrill, WordPress, cookies, copyright)',
                'relevant_context should list the keys from the related_context in that are found in the text (e.g., ["vic_grupo_social", "captura_metodo", "perp_tipo1"])',
                'Extractive summary in Spanish: copy exact spans; DO NOT paraphrase; preserve modality ("soñaba ser", "quería ser", "aspiraba a")',
                'If no relevant info, relevant_context=[] and summary=""',
                'NO APOLOGIES, NO FILLER TEXT'
            ]
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

    def _extract_summary_from_response(self, response: Any) -> str:
        """
        Parses the summary API response.

        Args:
            response (Any): The raw response object from the API call.

        Returns:
            str: The extracted summary text, or 'No relevant information found'.
        """
        content = response.choices[0].message.content
        parsed = json.loads(content)
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
        
        prompt_structure = {
            'input_text': summary,
            'question': question,
            'possible_values': possible_values,
            'instructions': [
                'OUTPUT FORMAT: Return ONLY {"evidence":"evidence", "result": "your_classification"}',
                'DO NOT ECHO THE INPUT, QUESTION, OR POSSIBLE_VALUES IN YOUR RESPONSE',
                f'Your result MUST be one of the possible_values: {possible_values}',
                'If no information is found about this label, return empty string like {"evidence": "no information found about this label", "result": ""}',
            ]
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
        parsed = json.loads(content)
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

    def summarize_text(self, text: str) -> str:
        """
        Summarizes the text extracting relevant context.

        Args:
            text (str): The cleaned input text.

        Returns:
            str: Extracted summary in Spanish, or 'No relevant information found'.
        """
        if not text:
            return ""
        
        kwargs = self._get_summary_args(text)
        try:
            response = self.client.chat.completions.create(**kwargs)
            return self._extract_summary_from_response(response)
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return 'No relevant information found'

    def classify_summary(self, summary: str, key: str) -> str:
        """
        Classifies the summary for a single taxonomy key.

        Args:
            summary (str): The summarized text to classify.
            key (str): The taxonomy key to classify against (e.g., 'vic_grupo_social').

        Returns:
            str: The classification result for this key.
        """
        kwargs = self._get_classification_args(summary, key)
        if kwargs is None:
            return 'No information'

        try:
            response = self.client.chat.completions.create(**kwargs)
            return self._extract_classification_from_response(response)
        except Exception as e:
            self.logger.error(f"Classification failed for {key}: {e}")
            return 'No information'


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

    async def summarize_text(self, text: str) -> str:
        """
        Summarizes the text extracting relevant context (Async).

        Args:
            text (str): The cleaned input text.

        Returns:
            str: Extracted summary in Spanish, or 'No relevant information found'.
        """
        if not text:
            return ""
        
        kwargs = self._get_summary_args(text)
        try:
            response = await self.client.chat.completions.create(**kwargs)
            return self._extract_summary_from_response(response)
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return 'No relevant information found'

    async def classify_summary(self, summary: str, key: str) -> str:
        """
        Classifies the summary for a single taxonomy key (Async).

        Args:
            summary (str): The summarized text to classify.
            key (str): The taxonomy key to classify against (e.g., 'vic_grupo_social').

        Returns:
            str: The classification result for this key.
        """
        kwargs = self._get_classification_args(summary, key)
        if kwargs is None:
            return 'No information'

        try:
            response = await self.client.chat.completions.create(**kwargs)
            return self._extract_classification_from_response(response)
        except Exception as e:
            self.logger.error(f"Classification failed for {key}: {e}")
            return 'No information'
