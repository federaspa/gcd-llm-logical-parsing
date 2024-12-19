from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama
import json
from typing import List, Tuple
from pushover import Pushover
import dotenv
import os
import logging
import sys
from timeout_decorator import timeout
from utils.prompters import FOL_Prompter, SAT_Prompter, LP_Prompter
from datetime import datetime
import numpy as np

dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
USER_KEY = os.getenv("USER_KEY")


def get_logger(script_name, debug:bool = False):
    
    
    current_datetime = datetime.now().strftime("%d%m%Y_%H%M%S")

    # Create a logger
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Create file handler which logs even debug messages
    log_file_name = f"logs/{script_name}_{current_datetime}.log"
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file_name

def calculate_perplexity(logprobs):
    return float(np.exp(-np.mean(logprobs['token_logprobs'])))



class InvalidJsonError(Exception):
    pass
    
class PromptGenerator:
    def __init__(self, config):
        self.script_config = config
        self.type = self._get_type()
        self._load_templates()
        self.prompter = self._get_prompter()

    def _get_type(self) -> str:
        types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'PARARULE': 'FOL',
            'LogicNLI': 'FOL',
            'LogicNLI_symbolic': 'FOL',
            'ProntoQA': 'FOL',
            'ProofWriter': 'FOL',
            'AR-LSAT': 'SAT'
        }
        return types[self.script_config.dataset_name]
    
    def _get_prompter(self):
        prompters = {
            'FOL': FOL_Prompter,
            'SAT': SAT_Prompter,
            'LP': LP_Prompter
        }
        
        prompter = prompters[self.type](self.script_config, self.templates)
        return prompter
    
    def _get_prompt_template(self):
        
        prompt_templates = {
            'gemma': 'prompts/prompt_templates/gemma.txt',
            'llama': 'prompts/prompt_templates/llama.txt',
            'mistral': 'prompts/prompt_templates/mistral.txt',
            'ministral': 'prompts/prompt_templates/ministral.txt',
            'qwen': 'prompts/prompt_templates/qwen.txt',
            'qwen2.5': 'prompts/prompt_templates/qwen.txt',
        }
        
        model_base = self.script_config.model_name.split('-')[0]
        
        return prompt_templates[model_base]

    def _load_templates(self):
        templates = {
            'unconstrained': f'./prompts/conversion/{self.script_config.dataset_name}/unconstrained.txt',
            'json': f'./prompts/conversion/{self.script_config.dataset_name}/json.txt',
            'constrained': f'./prompts/conversion/{self.script_config.dataset_name}/constrained.txt',
            'constructs': f'./prompts/conversion/{self.script_config.dataset_name}/constructs.txt',
            'prompt_template': self._get_prompt_template(),
            'json_grammar': f'./LLMs/grammars/json.gbnf',
            'construncts_grammar': f'./LLMs/grammars/{self.type}_constructs.gbnf',
            'constrained_grammar': f'./LLMs/grammars/{self.type}_constrained.gbnf'
            }

        self.templates = {}
        for key, path in templates.items():
            with open(path, 'r') as f:
                content = f.read()
                if key in ['json', 'unconstrained', 'constrained', 'constructs']:
                    with open(templates['prompt_template'], 'r') as pt:
                        prompt_template = pt.read()
                        content = prompt_template.replace('[[user]]', content)
                self.templates[key] = content
                
                
class OSModel(PromptGenerator):   
    def __init__(self, script_config, default_model_config:dict, llama_cpp_config:dict):
        """
        Initialize model with config parameters.
        """
        
        self.script_config = script_config
        
        super().__init__(self.script_config)
        
        self.default_model_config = default_model_config
        # pp(llama_cpp_config)
        # llama_cpp_config = {"model_path": llama_cpp_config['model_path'], 'n_ctx': 2048}
        
        self.llm = Llama(
            f16_kv=True,
            logits_all=True,
            **llama_cpp_config
        )
        
        self.unconstrained_generator = timeout(seconds=self.script_config.timeout)(self._unconstrained_generator_base)
        self.json_generator = timeout(seconds=self.script_config.timeout)(self._json_generator_base)
        self.constrained_generator = timeout(seconds=self.script_config.timeout)(self._constrained_generator_base)

    def invoke(self, prompt: str, model_config: dict = None, raw_grammar: str = None):
        """
        Generate completion with config parameters.
        """
        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        model_config = self.default_model_config if not model_config else model_config
        
        response =  self.llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            logprobs=1,
            **model_config
        )
        
        return response
    
    def _get_grammar(self, sample: dict, twosteps: bool) -> List[str]:
        """Extract constructs from natural language using LLM"""
        
        if twosteps:
            
            user = self.prompter.extract_constructs(sample)
            response = self.invoke(
                prompt=user,
                raw_grammar=self.templates['json_grammar'],
            )
            
            content = response['choices'][0]['text']
            constructs = json.loads(content)
        
        else:
            constructs = {}
        
        grammar = self.prompter.build_grammar(constructs)

        return grammar

    def _unconstrained_generator_base(self, sample: dict) -> Tuple[str, float]:
        user = self.prompter.unconstrained(sample=sample)
        response = self.invoke(
            prompt=user
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        return content, perplexity

    def _json_generator_base(self, sample: dict) -> Tuple[str, float]:
        user = self.prompter.json(sample=sample)
        response = self.invoke(
            prompt=user,
            raw_grammar=self.templates['json_grammar']
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        # try:
        #     content = json.loads(content)
        # except json.JSONDecodeError:
        #     raise InvalidJsonError(content)
        
        return content, perplexity
    
    def _constrained_generator_base(self, sample: str, twosteps: bool) -> Tuple[str, float]:
        sample_grammar = self._get_grammar(sample, twosteps)
        
        user = self.prompter.constrained(sample)
        response = self.invoke(
            prompt=user,
            raw_grammar=sample_grammar
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        # try:
        #     content = json.loads(content)
        # except json.JSONDecodeError:
        #     raise InvalidJsonError(content)
        
        return content, perplexity