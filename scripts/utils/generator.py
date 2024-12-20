from utils.template_manager import TemplateManager
from utils.prompters import FOL_Prompter
from utils.model import LLMModel
import json
import numpy as np
from timeout_decorator import timeout
from typing import Tuple

class Generator:
    def __init__(self, config):
        self.config = config
        self.template_manager = TemplateManager(config)
        self.prompter = self._get_prompter()
        self.model = None  # Initialize in setup_model
        
        self.generate_unconstrained = timeout(seconds=config.timeout)(self._generate_unconstrained_base)
        self.generate_json = timeout(seconds=config.timeout)(self._generate_json_base)
        self.generate_constrained = timeout(seconds=config.timeout)(self._generate_constrained_base)
        
    def _get_prompter(self):
        prompters = {
            'FOL': FOL_Prompter,
        }
        return prompters[self.template_manager.type](self.config, self.template_manager.templates)

    def setup_model(self, default_model_config: dict, llama_cpp_config: dict):
        self.model = LLMModel(self.config, default_model_config, llama_cpp_config)

    def _get_grammar(self, sample: dict, twosteps: bool) -> str:
        if not twosteps:
            return self.prompter.build_grammar({})

        response = self.model.invoke(
            prompt=self.prompter.extract_constructs(sample),
            raw_grammar=self.template_manager.templates['json_grammar'],
        )
        
        content = response['choices'][0]['text']
        constructs = json.loads(content)
        return self.prompter.build_grammar(constructs)

    @staticmethod
    def calculate_perplexity(logprobs) -> float:
        return float(np.exp(-np.mean(logprobs['token_logprobs'])))

    def _generate_unconstrained_base(self, sample: dict) -> Tuple[str, float]:
        response = self.model.invoke(prompt=self.prompter.unconstrained(sample))
        return response['choices'][0]['text'], self.calculate_perplexity(response['choices'][0]['logprobs'])

    def _generate_json_base(self, sample: dict) -> Tuple[str, float]:
        response = self.model.invoke(
            prompt=self.prompter.json(sample),
            raw_grammar=self.template_manager.templates['json_grammar']
        )
        return response['choices'][0]['text'], self.calculate_perplexity(response['choices'][0]['logprobs'])

    def _generate_constrained_base(self, sample: dict, twosteps: bool) -> Tuple[str, float]:
        sample_grammar = self._get_grammar(sample, twosteps)
        response = self.model.invoke(
            prompt=self.prompter.constrained(sample),
            raw_grammar=sample_grammar
        )
        return response['choices'][0]['text'], self.calculate_perplexity(response['choices'][0]['logprobs'])

    # def generate_unconstrained(self, sample: dict) -> Tuple[str, float]:
    #     return self._generate_unconstrained_base(sample)

    # def generate_json(self, sample: dict) -> Tuple[str, float]:
    #     return self._generate_json_base(sample)

    # def generate_constrained(self, sample: dict, twosteps: bool) -> Tuple[str, float]:
    #     return self._generate_constrained_base(sample, twosteps)