from typing import Optional
from llama_cpp import LlamaGrammar, Llama
from utils.template_manager import TemplateManager
from utils.prompters import FOL_Prompter, SIN_Prompter, LD_Prompter
import json
import numpy as np
from timeout_decorator import timeout
from abc import abstractmethod
from typing import Tuple

class BaseLLM:
    def __init__(self, config, default_model_config: dict, llama_cpp_config: dict):
        self.config = config
        self.default_model_config = default_model_config
        self.llm = Llama(f16_kv=True, logits_all=True, **llama_cpp_config)
        
    def invoke(self, prompt: str, model_config: Optional[dict] = None, raw_grammar: Optional[str] = None) -> dict:
        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        model_config = model_config or self.default_model_config
        
        return self.llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            logprobs=1,
            **model_config
        )
        
class MaxTokensException(Exception):
    def __init__(self, message="Maximum token limit reached"):
        self.message = message
        super().__init__(self.message)

class Model:
    def __init__(self, config):
        self.config = config
        self.template_manager = TemplateManager(config)
        self.prompter = self._get_prompter()
        self.model = None  # Initialize in setup_model
        
        self.generate_problem = timeout(seconds=config.timeout)(self._generate_base)
        
        
    def _get_prompter(self):
        prompters = {
            'FOL': FOL_Prompter,
            'SIN': SIN_Prompter,
            'LD': LD_Prompter
        }
        return prompters[self.template_manager.type](self.config)

    def setup_model(self, default_model_config: dict, llama_cpp_config: dict):
        self.model = BaseLLM(self.config, default_model_config, llama_cpp_config)
        
    @abstractmethod
    def generate_problem(self, sample: dict, constraint_type: str) -> Tuple[str, float]:
        pass

    @staticmethod
    def calculate_perplexity(logprobs) -> float:
        return float(np.exp(-np.mean(logprobs['token_logprobs'])))
    
    
    def _generate_base(self, sample: dict, constraint_type: str) -> Tuple[str, float]:
        prompt = self.prompter.format_prompt(sample)
        raw_grammar = self.prompter.build_grammar(constraint_type)
        
        response = self.model.invoke(
            prompt=prompt,
            raw_grammar=raw_grammar
        )
            
        return response