import json
import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError

from utils import OSModel, send_notification, get_logger, calculate_perplexity
from prompters import FOL_Prompter, SAT_Prompter, LP_Prompter

import traceback

@dataclass
class Config:
    sketcher_name: str
    dataset_name: str
    data_path: str = './data'
    split: str = 'dev'
    models_path: str = '/data/users/fraspant/LLMs'
    save_path: str = './outputs/logic_problems'
    n_gpu_layers: int = 0
    verbose: bool = False

script_name = Path(__file__).stem
logger, log_file_name = get_logger(script_name)

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
            'structured_user': f'./prompts/conversion/{self.config.dataset_name}/structured.txt',
            'unstructured_user': f'./prompts/conversion/{self.config.dataset_name}/unstructured.txt',
            'prompt_template': 'prompts/prompt_templates/gemma.txt' if 'gemma' in self.config.sketcher_name else 'prompts/prompt_templates/llama.txt',
            'json_grammar': './LLMs/grammars/json.gbnf'
        }

        # Load and process templates
        self.templates = {}
        for key, path in templates.items():
            with open(path, 'r') as f:
                content = f.read()
                if key in ['structured_user', 'unstructured_user']:
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
            verbose=config.verbose
        )

    def _load_raw_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    @timeout(seconds=180)
    def _unstructured_generator(self, sample: dict) -> Tuple[str, float]:
        user = self.prompter.unstructured(sample=sample)
        response = self.sketcher_api.invoke(prompt=user)
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity

    @timeout(seconds=180)
    def _structured_generator(self, unstructured: str) -> dict:
        user = self.prompter.structured(unstructured)
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.templates['json_grammar']
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return json.loads(content), perplexity

    def run(self):
        raw_dataset = self._load_raw_dataset()
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_file = save_path / f'{self.config.dataset_name}_{self.config.split}_{self.config.sketcher_name}.json'
        
        outputs = []
        if save_file.exists():
            with open(save_file, 'r') as f:
                outputs = json.load(f)

            existing_ids = {s["id"] for s in outputs}
            raw_dataset = [s for s in raw_dataset if s["id"] not in existing_ids]

        logger.info(f"{len(outputs)} already exist.\nLoaded {len(raw_dataset)} examples from {self.config.split} split.")

        for i, sample in enumerate(tqdm(raw_dataset)):
            try:
                try:
                    unstructured, perplexity = self._unstructured_generator(sample)
                except TimeoutError:
                    logger.warning(f"Timeout occurred during unstructured generation for sample {sample['id']}")
                    continue
                    
                try:
                    logic_problem, struct_perplexity = self._structured_generator(unstructured)
                except TimeoutError:
                    logger.warning(f"Timeout occurred during structured generation for sample {sample['id']}")
                    continue
                
                logic_problem['perplexity'] = (perplexity, struct_perplexity)
                    
                if i % 20 == 0:
                    logger.debug(logic_problem)
                
                output = {
                    'id': sample['id'], 
                    'nl_problem': {
                        'nl_rules': sample['context'],
                        'nl_conc': sample['question'],
                        'options': sample.get('options', [])
                    },
                    'answer': sample['answer'],
                    'logic_problem': logic_problem
                }
                
                outputs.append(output)
                
            except Exception as e:
                error_message = f"An error occurred for sample {sample['id']}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                send_notification(error_message, f"{script_name} sample error")
                logger.error(error_message)
            
            with open(save_file, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

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
    parser.add_argument('--n-gpu-layers', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == '__main__':
    config = parse_args()
    
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Sketcher: {config.sketcher_name}")
    logger.info(f"Split: {config.split}")
    logger.info(f"Save path: {config.save_path}")
    
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