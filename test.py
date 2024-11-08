import json
import os

import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm.autonotebook import tqdm
from timeout_decorator import timeout, TimeoutError
from datetime import datetime

from models.utils import OSModel, send_notification, get_logger, calculate_perplexity
from models.prompters import FOL_Prompter, SAT_Prompter, LP_Prompter

import traceback
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.environ["CUDA_VISIBLE_DEVICES"])


@dataclass
class Config:
    sketcher_name: str
    dataset_name: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    n_gpu_layers: int 
    n_ctx: int
    n_threads: int
    stop_time: str|None
    timeout_seconds: int|None
    verbose: bool = False

class PromptGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.type = self._get_type()
        self._load_templates()
        self.prompter = self._get_prompter()

    def _get_type(self) -> str:
        types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'LogicNLI': 'FOL',
            'ProntoQA': 'LP',
            'ProofWriter': 'LP',
            'AR-LSAT': 'SAT'
        }
        return types[self.config.dataset_name]
    
    def _get_prompter(self):
        prompters = {
            'FOL': FOL_Prompter,
            'SAT': SAT_Prompter,
            'LP': LP_Prompter
        }
        
        prompter = prompters[self.type](self.config, self.templates)
        return prompter

    def _load_templates(self):
        templates = {
            'json_user': f'./prompts/conversion/{self.config.dataset_name}/json.txt',
            'constrained_user': f'./prompts/conversion/{self.config.dataset_name}/constrained.txt',
            'unconstrained_user': f'./prompts/conversion/{self.config.dataset_name}/unconstrained.txt',
            'prompt_template': 'prompts/prompt_templates/gemma.txt' if 'gemma' in self.config.sketcher_name else 'prompts/prompt_templates/llama.txt',
            'json_grammar': './LLMs/grammars/json.gbnf',
            'constrained_grammar': f'./LLMs/grammars/{self.type}_constrained.gbnf'}

        self.templates = {}
        for key, path in templates.items():
            with open(path, 'r') as f:
                content = f.read()
                if key in ['json_user', 'unconstrained_user']:
                    with open(templates['prompt_template'], 'r') as pt:
                        prompt_template = pt.read()
                        content = prompt_template.replace('[[user]]', content)
                self.templates[key] = content

class LogicProgramGenerator(PromptGenerator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.sketcher_path = Path(config.models_path) / f"{config.sketcher_name}.gguf"
        assert self.sketcher_path.exists(), f"Model path does not exist: {self.sketcher_path}"
        
        self.sketcher_api = OSModel(
            model_path=str(self.sketcher_path), 
            n_gpu_layers=config.n_gpu_layers,
            n_threads = config.n_threads,
            n_ctx=config.n_ctx,
            verbose=config.verbose
        )
        
        self._constrained_generator = timeout(seconds=config.timeout_seconds)(self._constrained_generator_base)
    
    def _load_raw_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def _constrained_generator_base(self, sample: str) -> dict:
        user = self.prompter.constrained(sample)
        
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.templates['constrained_grammar'],
            temperature=0.5,
            top_p = 1,
            top_k=10,
            min_p=0.1,
            tfs_z=1,
            repeat_penalty=1
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])


        print(content)
        print(perplexity)

        return content, perplexity
    

    def run(self):
        raw_dataset = self._load_raw_dataset()

        for i, sample in enumerate(tqdm(raw_dataset, total=len(raw_dataset))):
            
            try:
                input('Continue?')
                
                nl_problem = sample
                self._constrained_generator(nl_problem)
            except KeyboardInterrupt:
                sys.exit(0)
                
            except:
                traceback.format_exc()
            
def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--n-gpu-layers', type=int, default=0)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-ctx', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--timeout-seconds', type=int, default=None, help='Timeout in seconds for generation operations')
    
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == '__main__':
    config = parse_args()
    
    generator = LogicProgramGenerator(config)
    generator.run()