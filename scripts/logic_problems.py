import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import yaml
import traceback

from utils.generator import PromptGenerator
from utils.logger import get_logger

@dataclass
class ScriptConfig:
    model_name: str
    dataset_name: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    timeout: Optional[int]
    stop_time: Optional[str]
    two_steps: bool = False
    force_unconstrained: bool = False
    force_constrained: bool = False
    force_json: bool = False
    debug: bool = False

class LogicProgramRunner:
    def __init__(self, script_config: ScriptConfig, model_config: dict, llama_cpp_config: dict):
        self.config = script_config
        self.generator = PromptGenerator(script_config)
        self.generator.setup_model(model_config, llama_cpp_config)
        
        self.stop_time = None
        if self.config.stop_time:
            try:
                self.stop_time = datetime.strptime(self.config.stop_time, "%d-%m-%y:%H-%M-%S")
            except ValueError as e:
                raise ValueError(f"Invalid stop_time format. Use dd-mm-yy:hh-mm-ss. Error: {str(e)}")

    def _check_time_limit(self) -> bool:
        """Check if we've reached the stop time"""
        if self.stop_time and datetime.now() >= self.stop_time:
            logger.info(f"Stop time {self.stop_time} reached. Saving progress and exiting...")
            return True
        return False
    
    def _load_raw_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            return json.load(f)
    
    def _skip_existing(self, save_file: Path, raw_dataset: List[Dict]) -> Tuple[List[Dict], Dict, List]:
        outputs = {}
        existing_ids = []
        
        if save_file.exists():
            with open(save_file, 'r') as f:
                saved_data = json.load(f)
                outputs = {sample['id']: sample for sample in saved_data}
                existing_ids = list(outputs.keys())
                
        return raw_dataset, outputs, existing_ids

    def _generate_problem(self, nl_problem: dict, sample_id: int, problem_type: str, pbar: tqdm) -> Tuple[Optional[dict], Optional[str]]:
        """Generate logic problem for a sample"""
        pbar.set_description_str(f"Generating {problem_type} problem {sample_id}")
        
        try:
            if problem_type == 'unconstrained':
                logic_problem_str, perplexity = self.generator.generate_unconstrained(nl_problem)
            elif problem_type == 'json':
                logic_problem_str, perplexity = self.generator.generate_json(nl_problem)
            elif problem_type == 'constrained':
                logic_problem_str, perplexity = self.generator.generate_constrained(nl_problem, self.config.two_steps)
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            
            return {
                'raw': logic_problem_str,
                'perplexity': perplexity
            }, None
            
        except TimeoutError:
            logger.error(f"{problem_type.capitalize()}: Timeout error for sample {sample_id}")
            return None, "timeout"
        except Exception as e:
            logger.error(f"{problem_type.capitalize()}: An error occurred for sample {sample_id}: {str(e)}")
            logger.debug(f"Traceback:\n{traceback.format_exc()}")
            return None, str(e)
        
    def _prepare_save_file(self):
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_name = re.sub(r'-\d+-of-\d+', '', self.config.model_name)

        return save_path / f'{self.config.dataset_name}_{self.config.split}_{model_name}.json'
        

    def run(self):
        # Setup paths and load data
        raw_dataset = self._load_raw_dataset()

        save_file = self._prepare_save_file()
        raw_dataset, outputs, existing_ids = self._skip_existing(save_file, raw_dataset)
        
        logger.info(f"Loaded {len(raw_dataset)} examples from {self.config.split} split.")

        # Process samples
        for i, sample in enumerate(pbar := tqdm(raw_dataset, total=len(raw_dataset), bar_format='{desc}')):
            if self._check_time_limit():
                break
            
            # Get or create problem
            if sample['id'] in existing_ids:
                sample = outputs[sample['id']]
                nl_problem = sample['nl_problem']
            else:
                nl_problem = sample
            
            # Generate different versions of the problem
            output = {
                'id': sample['id'],
                'nl_problem': {
                    'context': nl_problem['context'],
                    'question': nl_problem['question'],
                    'options': nl_problem.get('options', []),
                    'answer': nl_problem['answer']
                }
            }

            # Generate unconstrained version
            if 'logic_problem' not in sample or self.config.force_unconstrained:
                logic_problem, _ = self._generate_problem(nl_problem, sample["id"], "unconstrained", pbar)
                if logic_problem:
                    output['logic_problem'] = logic_problem
            else:
                output['logic_problem'] = sample['logic_problem']

            # Generate JSON version
            if 'logic_problem_json' not in sample or self.config.force_json:
                logic_problem_json, _ = self._generate_problem(nl_problem, sample["id"], "json", pbar)
                if logic_problem_json:
                    output['logic_problem_json'] = logic_problem_json
            else:
                output['logic_problem_json'] = sample['logic_problem_json']

            # Generate constrained version
            if 'logic_problem_gcd' not in sample or self.config.force_constrained:
                logic_problem_gcd, _ = self._generate_problem(nl_problem, sample["id"], "constrained", pbar)
                if logic_problem_gcd:
                    output['logic_problem_gcd'] = logic_problem_gcd
            else:
                output['logic_problem_gcd'] = sample['logic_problem_gcd']

            # Save progress
            outputs[sample['id']] = output
            with open(save_file, 'w') as f:
                json.dump(list(outputs.values()), f, indent=2, ensure_ascii=False)

            # Debug logging
            if i % 20 == 0:
                for key in ['logic_problem', 'logic_problem_json', 'logic_problem_gcd']:
                    if key in output:
                        logger.debug(f"{key}: {output[key]}")

        if self._check_time_limit():
            logger.info("Script stopped due to reaching stop time")
        else:
            logger.info(f"Generated {len(outputs)} examples.")

def parse_args() -> ScriptConfig:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--two-steps', action='store_true', help='Extract predicates in first step')
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--timeout', type=float, default=None, help='Timeout in seconds for generation operations')
    parser.add_argument('--force-unconstrained', action='store_true')
    parser.add_argument('--force-constrained', action='store_true')
    parser.add_argument('--force-json', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    return ScriptConfig(**vars(parser.parse_args()))

def get_configs(script_config: ScriptConfig) -> Tuple[ScriptConfig, dict, dict]:
    
    model_name = re.sub(r'-\d+-of-\d+', '', script_config.model_name)
    
    # Load model-specific configuration
    sketcher_config_path = Path('configs/models') / f"{model_name}.yml"
    with open(sketcher_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # Load general LlamaCpp generation configuration
    llamacpp_config_path = Path('configs') / 'llamacpp.yml'
    with open(llamacpp_config_path, 'r') as f:
        llama_cpp_config = yaml.safe_load(f)
    
    # Add model path to model config
    llama_cpp_config['model_path'] = str(Path(script_config.models_path) / f"{script_config.model_name}.gguf")
    
    # Verify model path exists
    assert Path(llama_cpp_config['model_path']).exists()
    
    return script_config, model_config, llama_cpp_config

if __name__ == '__main__':
    args = parse_args()
    
    script_name = Path(__file__).stem
    logger, log_file_name = get_logger(script_name, args.debug)
    
    try:
        script_config, model_config, llama_cpp_config = get_configs(args)
        runner = LogicProgramRunner(script_config, model_config, llama_cpp_config)
        runner.run()
        
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        os.remove(f"./{log_file_name}")
        sys.exit(0)
        
    except Exception as e:
        error_message = f"A fatal error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_message)
        sys.exit(0)
        
    logger.info("Finished Successfully")