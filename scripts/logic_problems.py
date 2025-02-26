import json
import os
import sys
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
import yaml
import traceback

from utils.generator import Generator
from utils.logger import get_logger
from utils.notifier import PushoverNotifier

@dataclass
class ScriptConfig:
    model_name: str
    dataset_name: str
    shots_number: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    timeout: int
    starting_sample: int
    verbose: bool
    n_gpu_layers: int
    start_time: Optional[str]
    stop_time: Optional[str]
    force_unconstrained: Optional[bool]
    force_constrained: Optional[bool]
    force_json: Optional[bool]

class LogicProgramRunner:
    def __init__(self, script_config: ScriptConfig, model_config: dict, llama_cpp_config: dict, logger):
        self.config = script_config
        self.generator = Generator(script_config)
        self.generator.setup_model(model_config, llama_cpp_config)
        self.save_file = self._prepare_save_file()
        
        
        self.logger = logger
        
        self.stop_time = None   
        if self.config.stop_time:
            try:
                self.stop_time = datetime.strptime(self.config.stop_time, "%d-%m-%y:%H-%M-%S")
            except ValueError as e:
                raise ValueError(f"Invalid stop_time format. Use dd-mm-yy:hh-mm-ss. Error: {str(e)}")

    def _check_time_limit(self) -> bool:
        """Check if we've reached the stop time"""
        if self.stop_time and datetime.now() >= self.stop_time:
            self.logger.info(f"Stop time {self.stop_time} reached. Saving progress and exiting...")
            return True
        return False
    
    def _prepare_save_file(self):
        save_path = Path(os.path.join(self.config.save_path, self.config.shots_number, 'logic_problems'))
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_name = re.sub(r'-\d+-of-\d+', '', self.config.model_name)

        return save_path / f'{self.config.dataset_name}_{self.config.split}_{model_name}.json'
    
    def _load_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
            
        cut_dataset = raw_dataset[self.config.starting_sample:]
        
        with open(os.path.join(self.config.data_path, self.config.dataset_name, 'sampled_ids.json')) as f:
            sampled_ids = json.load(f)
            
        sampled_dataset = [s for s in cut_dataset if s['id'] in sampled_ids]

        outputs = {}
        existing_ids = []
        
        if self.save_file.exists():
            with open(self.save_file, 'r') as f:
                saved_data = json.load(f)
                
            outputs = {sample['id']: sample for sample in saved_data}
            existing_ids = list(outputs.keys())
                
        return sampled_dataset, outputs, existing_ids

    def _generate_problem(self, nl_problem: dict, sample_id: int, constraint_type: str) -> Dict:
        """Generate logic problem for a sample"""
        
        assert constraint_type in ['unconstrained', 'json', 'constrained'], ValueError(f"Invalid problem type: {constraint_type}")
        
        start_time = datetime.now()
        
        try:
            response = self.generator.generate_problem(nl_problem, constraint_type)
            logic_problem_str = response['choices'][0]['text']
            perplexity = self.generator.calculate_perplexity(response['choices'][0]['logprobs'])
            
            duration = (datetime.now() - start_time).total_seconds()
            error_message = None
            
            if response['choices'][0]['finish_reason'] == 'length':
                self.logger.error(f"{constraint_type.capitalize()}: Max tokens error for sample {sample_id}")
                error_message = "max_tokens"
            
        except TimeoutError:
            self.logger.error(f"{constraint_type.capitalize()}: Timeout error for sample {sample_id}")
            logic_problem_str, perplexity, duration = None, None, self.config.timeout
            error_message = "timeout"

        except Exception as e:
            self.logger.error(f"{constraint_type.capitalize()}: An error occurred for sample {sample_id}: {str(e)}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            logic_problem_str, perplexity, duration = None, None, None
            error_message = str(e)
        
        return {
            'raw': logic_problem_str,
            'perplexity': perplexity,
            'generation_time': duration,
            'error_message': error_message
        }
        

        

    def run(self):
        
        raw_dataset, outputs, existing_ids = self._load_dataset()
        self.logger.info(f"Loaded {len(raw_dataset)} examples from {self.config.split} split.")

        # Process samples
        for i, sample in enumerate(pbar := tqdm(raw_dataset, total=len(raw_dataset), bar_format='{desc}: [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]]')):
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
                pbar.set_description_str(f"Generating unconstrained problem {sample['id']} ({i}/{len(raw_dataset)})")
                logic_problem = self._generate_problem(nl_problem, sample["id"], "unconstrained")
                output['logic_problem'] = logic_problem
            else:
                output['logic_problem'] = sample['logic_problem']

            # Generate JSON version
            if self.config.shots_number != 'baseline':
                if 'logic_problem_json' not in sample or self.config.force_json:
                    pbar.set_description_str(f"Generating json problem {sample['id']} ({i}/{len(raw_dataset)})")
                    logic_problem_json = self._generate_problem(nl_problem, sample["id"], "json")
                    output['logic_problem_json'] = logic_problem_json
                else:
                    output['logic_problem_json'] = sample['logic_problem_json']

            # Generate constrained version
            if self.config.shots_number != 'baseline':
                if 'logic_problem_gcd' not in sample or self.config.force_constrained:
                    pbar.set_description_str(f"Generating constrained problem {sample['id']} ({i}/{len(raw_dataset)})")
                    logic_problem_gcd = self._generate_problem(nl_problem, sample["id"], "constrained")
                    output['logic_problem_gcd'] = logic_problem_gcd
                else:
                    output['logic_problem_gcd'] = sample['logic_problem_gcd']

            # Save progress
            outputs[sample['id']] = output
            with open(self.save_file, 'w') as f:
                json.dump(list(outputs.values()), f, indent=2, ensure_ascii=False)

            # Debug logging
            if i % 20 == 0:
                for key in ['logic_problem', 'logic_problem_json', 'logic_problem_gcd']:
                    if key in output:
                        self.logger.debug(f"{key}: {output[key]}")

        if self._check_time_limit():
            self.logger.info("Script stopped due to reaching stop time")
        else:
            self.logger.info(f"Generated {len(outputs)} examples.")

def get_configs(script_config: ScriptConfig) -> Tuple[ScriptConfig, dict, dict]:
    full_model_name = re.sub(r'-\d+-of-\d+', '', script_config.model_name)
    
    # Load model-specific configuration
    sketcher_config_path = Path('configs/models') / f"{full_model_name}.yml"
    with open(sketcher_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # Load general LlamaCpp generation configuration
    llamacpp_config_path = Path('configs') / 'llamacpp.yml'
    with open(llamacpp_config_path, 'r') as f:
        llama_cpp_config = yaml.safe_load(f)
    
    # Add model path to model config
    base_model_name = full_model_name.split('-')[0]
    model_dir = Path(script_config.models_path) / base_model_name
    
    # Check for multi-part model files
    model_pattern = f"{script_config.model_name}-*-of-*.gguf"
    model_parts = list(model_dir.glob(model_pattern))
    
    if model_parts:
        # Multi-part model - find first part
        model_parts.sort()  # Ensure proper ordering
        model_path = str(model_parts[0])
    else:
        # Single file model
        model_path = str(model_dir / f"{script_config.model_name}.gguf")
        
    # Verify model path exists
    assert Path(model_path).exists(), f'Model not found: {model_path}'
    
    llama_cpp_config['model_path'] = model_path
    llama_cpp_config['verbose'] = script_config.verbose
    llama_cpp_config['n_gpu_layers'] = script_config.n_gpu_layers
    llama_cpp_config['n_threads'] = 1 if script_config.n_gpu_layers == -1 else 12
    
    return script_config, model_config, llama_cpp_config

def parse_args() -> ScriptConfig:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--shots-number', type=str, default='2shots', choices=['0shots', '2shots', '5shots', 'baseline'])
    parser.add_argument('--n-gpu-layers', type=int, default=-1)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--data-path', type=str, default='/data/users/fraspant/GCLLM/data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--starting-sample', type=int, default=0)
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='/data/users/fraspant/GCLLM/outputs')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for generation operations')
    parser.add_argument('--start-time', default=None, type=str, help='Start time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--force-unconstrained', action='store_true')
    parser.add_argument('--force-constrained', action='store_true')
    parser.add_argument('--force-json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    return ScriptConfig(**vars(parser.parse_args()))

def extract_model_size(model_name):
    # Extract the size part (e.g., "8b" from "modelname2.0-8b-it")
    try:
        # Find the part that ends with 'b' and convert to number
        size_str = next(part for part in model_name.split('-') if part.endswith('b'))
        return int(size_str[:-1])  # Remove 'b' and convert to int
    except (StopIteration, ValueError):
        return 0  # Return 0 if pattern not found or conversion fails

if __name__ == '__main__':
    args = parse_args()

    script_name = Path(__file__).stem
    logger, log_file_name = get_logger(script_name, args.verbose)
    notifier = PushoverNotifier()
    retry_timeout = 300
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, "%d-%m-%y:%H-%M-%S")
        
        if datetime.now() < start_time:
            
            logger.info(f"Waiting to start script at {start_time}. Current time: {datetime.now()}, {round((start_time - datetime.now()).total_seconds()/3600)} hours ({round((start_time - datetime.now()).total_seconds()/60)} minutes) remaining")
        
        while datetime.now() < start_time:
            time.sleep((start_time - datetime.now()).total_seconds())
            
            
    if re.search(r'-\d+b', args.model_name, re.I):
        remaining_models = [args.model_name]
    else:
        remaining_models = [os.path.splitext(m)[0] for m in os.listdir('configs/models') if args.model_name in m]
        
    remaining_models = sorted(remaining_models, key=extract_model_size, reverse=True)
        
    last_attempt_times = {model: 0 for model in remaining_models} 
        
    logger.info(f'Script started with models: {remaining_models}')
    
    while remaining_models:
        current_time = time.time()
        
        for model_name in remaining_models:
            if current_time - last_attempt_times[model_name] < retry_timeout:
                continue
            
            try:
                logger.info(f"Running model: {model_name}")

                last_attempt_times[model_name] = current_time
                
                args.model_name = model_name
                script_config, model_config, llama_cpp_config = get_configs(args)
                runner = LogicProgramRunner(script_config, model_config, llama_cpp_config, logger)
                runner.run()
                logger.info(f"Finished Successfully: {model_name}")
                notifier.send(f"Finished Successfully: {model_name}", 'Info')
                
                remaining_models.remove(model_name)
                
            except ValueError as e:
                if "Failed to load model from file:" in str(e) or "Failed to create llama_context" in str(e):
                    logger.warning(f'Failed to load model {model_name}, retrying in {retry_timeout/60} minutes')
                    # time.sleep(retry_timeout)
                    continue
                else:
                    logger.error(f"ValueError for {model_name}: {str(e)}")
                    notifier.send(f"ValueError for {model_name}: {str(e)}", 'Error')
                    sys.exit(1) 
                
            except KeyboardInterrupt:
                logger.error("KeyboardInterrupt")
                os.remove(f"./{log_file_name}")
                sys.exit(0)
                
            except Exception as e:
                error_message = f"A fatal error occurred with {model_name}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logger.error(error_message)
                notifier.send(error_message, 'Error')

                sys.exit(1)
                
        if remaining_models:
            time.sleep(1)

logger.info("All models completed successfully")
notifier.send("All models completed successfully", 'Info')