from llama_cpp.llama import LlamaGrammar
from llama_cpp import Llama
import warnings
from pushover import Pushover
import dotenv
import os
import logging
import sys
from dataclasses import dataclass
from prompters import FOL_Prompter, SAT_Prompter, LP_Prompter
from datetime import datetime
import numpy as np

dotenv.load_dotenv()
API_KEY = os.getenv("API_KEY")
USER_KEY = os.getenv("USER_KEY")

def send_notification(message, title):

    po = Pushover(token=API_KEY)
    po.user(USER_KEY)

    msg = po.msg(message=message)

    msg.set("title", title)

    po.send(msg)
    
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

class OSModel:
    def __init__(self, config):
        """
        Initialize model with config parameters.
        Required config fields: model_path, n_gpu_layers, n_batch, n_ctx, n_threads
        """
        self.llm = Llama(
            model_path=config.model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_batch=getattr(config, 'n_batch', 512),
            n_threads=config.n_threads,
            n_ctx=config.n_ctx,
            f16_kv=True,
            verbose=config.verbose,
            logits_all=getattr(config, 'logits_all', True),
            **{k:v for k,v in vars(config).items() if k not in [
                'model_path', 'n_gpu_layers', 'n_batch', 'n_threads', 
                'n_ctx', 'verbose', 'logits_all'
            ]}
        )

    def invoke(self, prompt: str, raw_grammar: str = None, config = None):
        """
        Generate completion with config parameters.
        Required config fields: logprobs, max_tokens, top_p, top_k, min_p
        """
        grammar = LlamaGrammar.from_string(raw_grammar, verbose=False) if raw_grammar else None
        
        response =  self.llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            logprobs=getattr(config, 'logprobs', 1),
            **{k:v for k,v in vars(config).items() if k not in ['prompt', 'grammar']}
        )
        
        return response
    
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
            'LogicNLI': 'FOL',
            'LogicNLI_symbolic': 'FOL',
            'ProntoQA': 'LP',
            'ProofWriter': 'LP',
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
        
        if 'gemma' in self.script_config.sketcher_name:
            return 'prompts/prompt_templates/gemma.txt'  
        
        elif 'tinyllama' in self.script_config.sketcher_name:
            return 'prompts/prompt_templates/tinyllama.txt'  
            
        else:
            return 'prompts/prompt_templates/llama.txt'

    def _load_templates(self):
        templates = {
            'json': f'./prompts/conversion/{self.script_config.dataset_name}/json.txt',
            'constrained': f'./prompts/conversion/{self.script_config.dataset_name}/constrained.txt',
            'unconstrained': f'./prompts/conversion/{self.script_config.dataset_name}/unconstrained.txt',
            'predicates': f'./prompts/conversion/{self.script_config.dataset_name}/predicates.txt',
            'prompt_template': self._get_prompt_template(),
            'json_grammar': './LLMs/grammars/json.gbnf',
            'constrained_grammar': f'./LLMs/grammars/{self.type}_constrained.gbnf'}

        self.templates = {}
        for key, path in templates.items():
            with open(path, 'r') as f:
                content = f.read()
                if key in ['json', 'unconstrained', 'constrained', 'predicates']:
                    with open(templates['prompt_template'], 'r') as pt:
                        prompt_template = pt.read()
                        content = prompt_template.replace('[[user]]', content)
                self.templates[key] = content