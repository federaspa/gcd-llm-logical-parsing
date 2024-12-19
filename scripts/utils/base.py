from abc import ABC, abstractmethod
from typing import Dict

class BasePrompter(ABC):
    def __init__(self, config: dict, templates: Dict[str, str]):
        self.config = config
        self.templates = templates

    @abstractmethod
    def build_grammar(self, constructs: dict) -> str:
        pass

    def unconstrained(self, sample: Dict) -> str:
        return self._format_prompt('unconstrained', sample)
    
    def constrained(self, sample: Dict) -> str:
        return self._format_prompt('constrained', sample)
    
    def json(self, sample: Dict) -> str:
        return self._format_prompt('json', sample)
    
    def extract_constructs(self, sample: Dict) -> str:
        return self._format_prompt('constructs', sample)

    def _format_prompt(self, template_key: str, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.templates[template_key].replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)