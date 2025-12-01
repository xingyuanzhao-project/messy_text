import json
import re
import logging
from typing import Dict, Any


class MessyTextProcessor:
    """
    Pure text processor for summarization and classification.
    Contains only atomic processing methods. Orchestration belongs in caller.
    """
    
    def __init__(self, client: Any, config: Dict[str, Any], taxonomy: Dict[str, Any], logger: logging.Logger):
        """
        Args:
            client: The OpenAI/vLLM client instance.
            config: runtime settings (model name, tokens, etc).
            taxonomy: static definitions (context_definitions, label_options).
            logger: logger instance.
        """
        self.client = client
        self.config = config
        self.definitions = taxonomy['context_definitions']
        self.labels = taxonomy['label_options']
        self.logger = logger
        self.model_name = config['model']['name']

    def clean_text(self, text: str) -> str:
        """
        Applies regex cleaning to input text.
        Matches notebook logic exactly.
        """
        if not isinstance(text, str):
            return str(text)
            
        # Regex from opr_3.1
        # 1. Remove URLs, parenthetical metadata, emojis, date patterns like d/d
        text = re.sub(r'https?://\S+|\([^)]*/[^)]*\)|[\ue000-\uf8ff]|\b\d/\d\b', '', text)
        # 2. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def summarize_text(self, text: str) -> str:
        """
        Step 1: Summarizes the text extracting relevant context.
        Returns 'No relevant information found' if empty or failed.
        """
        if not text:
            return ""

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

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt_content}],
                temperature=self.config['processing']['temperature'],
                max_tokens=self.config['processing']['max_tokens_summary'],
                extra_body={"guided_json": {
                    "type": "object", 
                    "properties": {
                        "info_found": {"type": "string"}, 
                        "relevant_context": {"type": "array"}, 
                        "summary": {"type": "string"}
                    }, 
                    "required": ["info_found", "relevant_context", "summary"]
                }}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            summary = parsed.get('summary', 'No relevant information found')
            return summary if summary else 'No relevant information found'
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return 'No relevant information found'

    def classify_summary(self, summary: str, key: str) -> str:
        """
        Classifies the summary for a single taxonomy key.
        
        Args:
            summary: The summarized text to classify.
            key: The taxonomy key to classify against (e.g., 'vic_grupo_social').
        
        Returns:
            str: The classification result for this key.
        """
        if not summary or summary == 'No relevant information found':
            return 'No information'
        
        question = self.definitions.get(key, '')
        possible_values = self.labels.get(key, [])
        
        if not question:
            self.logger.error(f"Key '{key}' not found in definitions")
            return 'No information'
        
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

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt_content}],
                temperature=self.config['processing']['temperature'],
                max_tokens=self.config['processing']['max_tokens_classification'],
                extra_body={"guided_json": {
                    "type": "object", 
                    "properties": {
                        "evidence": {"type": "string"}, 
                        "result": {"type": "string"}
                    }, 
                    "required": ["evidence", "result"]
                }}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            result = parsed.get('result', 'No information')
            return result if result else 'No information'
            
        except Exception as e:
            self.logger.error(f"Classification failed for {key}: {e}")
            return 'No information'


