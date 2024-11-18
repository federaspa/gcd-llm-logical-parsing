import json
import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
from datetime import datetime

from utils import OSModel, send_notification, get_logger, calculate_perplexity
from prompters import FOL_Prompter, SAT_Prompter, LP_Prompter

import traceback

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
    max_tokens: int
    n_threads: int
    stop_time: str|None
    timeout_seconds: int|None
    force_unconstrained: bool = False
    force_constrained:bool = False
    debug: bool = False

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
            verbose=config.debug
        )
        
        # Convert stop_time string to datetime object if provided
        self.stop_time = None
        if config.stop_time:
            try:
                self.stop_time = datetime.strptime(config.stop_time, "%d-%m-%y:%H-%M-%S")
            except ValueError as e:
                raise ValueError(f"Invalid stop_time format. Use dd-mm-yy:hh-mm-ss. Error: {str(e)}")
        
        # Set up timeout decorator with config timeout
        self._unconstrained_generator = timeout(seconds=config.timeout_seconds)(self._unconstrained_generator_base)
        self._json_wrapper = timeout(seconds=config.timeout_seconds)(self._json_wrapper_base)
        self._constrained_generator = timeout(seconds=config.timeout_seconds)(self._constrained_generator_base)

    def _check_time_limit(self):
        """Check if we've reached the stop time"""
        if self.stop_time and datetime.now() >= self.stop_time:
            logger.info(f"Stop time {self.stop_time} reached. Saving progress and exiting...")
            return True
        return False
    
    def _load_raw_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def _unconstrained_generator_base(self, sample: dict) -> Tuple[str, float]:
        user = self.prompter.unconstrained(sample=sample)
        response = self.sketcher_api.invoke(
            prompt=user,
            max_tokens=config.max_tokens
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity

    def _json_wrapper_base(self, unconstrained: str) -> dict:
        user = self.prompter.json_wrap(unconstrained)
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.templates['json_grammar'],
            max_tokens=config.max_tokens
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return json.loads(content), perplexity
    
    def _constrained_generator_base(self, sample: str) -> dict:
        user = self.prompter.constrained(sample)
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.templates['constrained_grammar'],
            max_tokens=config.max_tokens,
            temperature=0.5,
            top_p = 1,
            top_k=10,
            min_p=0.1,
            tfs_z=1,
            repeat_penalty=1
        )
        
        content = response['choices'][0]['text']
        
        try:
            content = json.loads(content)
        except:
            raise Exception(content)
        
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])

        return content, perplexity
    
    def _skip_existing(self, save_file:Path, raw_dataset:List[Dict]):
        outputs = []
        existing_ids = []
        # existing_samples = []
        
        if save_file.exists():
            with open(save_file, 'r') as f:
                outputs = json.load(f)
                existing_ids = [s['id'] for s in outputs]
                
        outputs = {sample['id']: sample for sample in outputs}
        return raw_dataset, outputs, existing_ids

    def run(self):
        raw_dataset = self._load_raw_dataset()
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_file = save_path / f'{self.config.dataset_name}_{self.config.split}_{self.config.sketcher_name}.json'
        
        # raw_dataset, outputs, existing_samples, existing_ids = self._skip_existing(save_file=save_file, raw_dataset=raw_dataset)
        raw_dataset, outputs, existing_ids = self._skip_existing(save_file=save_file, raw_dataset=raw_dataset)
        
        raw_dataset = [s for s in raw_dataset if s['id']>112]
        
        logger.info(f"Loaded {len(raw_dataset)} examples from {self.config.split} split.")

        for i, sample in enumerate(pbar := tqdm(raw_dataset, total=len(raw_dataset), bar_format='{desc}')):
            if self._check_time_limit():
                break
        
            if sample['id'] in existing_ids:
                sample = outputs[sample['id']]              
                nl_problem = sample['nl_problem']
            else:
                nl_problem = sample
                       
            logic_problem = None
            logic_problem_gcd = None
                        
            try:
                if not 'logic_problem' in sample.keys() or config.force_unconstrained:
                    pbar.set_description_str("Generating unconstrained problem %s" % sample['id'])
                    unconstrained, perplexity = self._unconstrained_generator(nl_problem)
                    
                    pbar.set_description_str("Json wrapping problem %s" % sample['id'])
                    logic_problem, json_perplexity = self._json_wrapper(unconstrained)
                    
                    logic_problem['perplexity'] = (perplexity, json_perplexity)
                    
                    if i % 20 == 0:
                        logger.debug(unconstrained)
                        
                else:
                    logic_problem = sample['logic_problem']
                    
                    
                if i % 20 == 0:
                    logger.debug(logic_problem)
                        
                    
            except TimeoutError:
                logger.warning(f"Timeout occurred during unconstrained generation for sample {sample['id']}")

            except Exception as e:
                logger.error(f"An error occurred for sample {sample['id']}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                
            try:
                if not 'logic_problem_gcd' in sample.keys() or config.force_constrained:
                    pbar.set_description_str("Generating constrained problem %s" % sample['id'])
                    logic_problem_gcd, gcd_perplexity= self._constrained_generator(nl_problem)
                    
                    logic_problem_gcd['perplexity'] = gcd_perplexity
                else:
                    logic_problem_gcd = sample['logic_problem_gcd']
                
                if i % 20 == 0:
                    logger.debug(logic_problem_gcd)
                    
            except TimeoutError:
                logger.warning(f"Timeout occurred during constrained generation for sample {sample['id']}")
                
            except Exception as e:
                logger.error(f"An error occurred for sample {sample['id']}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
                
            output = {
                'id': sample['id'], 
                'nl_problem': {
                    'context': nl_problem['context'],
                    'question': nl_problem['question'],
                    'options': nl_problem.get('options', [])
                },
                'answer': sample['answer']
            }
            
            if logic_problem:
                output.update({'logic_problem': logic_problem})
            
            if logic_problem_gcd:
                output.update({'logic_problem_gcd': logic_problem_gcd})
            
            outputs[sample['id']] = output
            
            pbar.update()
            with open(save_file, 'w') as f:
                json.dump(list(outputs.values()), f, indent=2, ensure_ascii=False)

        if self._check_time_limit():
            logger.info("Script stopped due to reaching stop time")
        else:
            logger.info(f"Generated {len(outputs)} examples.")

def parse_args() -> Config:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--n-gpu-layers', type=int, default=-1)
    parser.add_argument('--n-threads', type=int, default=1)
    parser.add_argument('--n-ctx', type=int, default=5120)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--timeout-seconds', type=int, default=None, help='Timeout in seconds for generation operations')
    parser.add_argument('--force-unconstrained', action='store_true')
    parser.add_argument('--force-constrained', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == '__main__':
    config = parse_args()
    
    script_name = Path(__file__).stem
    logger, log_file_name = get_logger(script_name, config.debug)
    
    
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Sketcher: {config.sketcher_name}")
    logger.info(f"Split: {config.split}")
    logger.info(f"Save path: {config.save_path}")
    logger.info(f"Threads: {config.n_threads}")
    logger.info(f"GPU layers: {config.n_gpu_layers}")
    logger.info(f"Max tokens: {config.max_tokens}")
    logger.info(f"Force unconstrained: {config.force_unconstrained}")
    logger.info(f"Force constrained: {config.force_constrained}")
    logger.info(f"Stop time: {config.stop_time}")
    logger.info(f"Operation timeout: {config.timeout_seconds} seconds")
    
    try:
        generator = LogicProgramGenerator(config)
        generator.run()
        
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        os.remove(f"./{log_file_name}")
        sys.exit(0)
        
    except Exception as e:
        error_message = f"A fatal error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        send_notification(error_message, f"{script_name} fatal error")
        logger.error(error_message)
        sys.exit(0)
        
    logger.info("Finished Successfully")
    send_notification("Yippiee!", f"{script_name} finished successfully")