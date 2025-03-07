import json
import os
import sys
import re
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import time
import yaml
import traceback

from utils.model import Model
from utils.logging import PushoverNotifier, get_logger
from utils.utils import get_configs, get_models, parse_args, ScriptConfig

class LogicProgramRunner:
    def __init__(self, script_config: ScriptConfig, model_config: dict, llama_cpp_config: dict, logger):
        self.config = script_config
        self.model_config = model_config
        self.llama_cpp_config = llama_cpp_config
        self.save_file = self._prepare_save_file()
        
        self.logger = logger

    def _load_model(self):
        self.generator = Model(self.config)
        self.generator.setup_model(self.model_config, self.llama_cpp_config)

    def _prepare_save_file(self):
        
        save_path = Path('./debug2')
        
        save_path.mkdir(parents=True, exist_ok=True)

        return save_path / f'{self.config.dataset_name}_{self.config.split}_{model_name}.json'
    
    def _load_dataset(self) -> List[dict]:
        dataset_path = Path(self.config.data_path) / self.config.dataset_name / f'{self.config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
            
        outputs = {}
        existing_ids = []
                
        return raw_dataset, outputs, existing_ids

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
        
        if len(raw_dataset):
            print("Loading model...")
            self._load_model()

        # Process samples
        for i, sample in enumerate(pbar := tqdm(raw_dataset, total=len(raw_dataset), bar_format='{desc}: [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]]')):
            
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
            
            # Generate constrained version
            pbar.set_description_str(f"Generating constrained problem {sample['id']} ({i}/{len(raw_dataset)})")
            logic_problem_gcd = self._generate_problem(nl_problem, sample["id"], "constrained")
            output['logic_problem_gcd'] = logic_problem_gcd

            # Save progress
            outputs[sample['id']] = output
            with open(self.save_file, 'w') as f:
                json.dump(list(outputs.values()), f, indent=2, ensure_ascii=False)

            for key in ['logic_problem', 'logic_problem_json', 'logic_problem_gcd']:
                if key in output:
                    self.logger.debug(f"{key}: {output[key]}")

        self.logger.info(f"Generated {len(outputs)} examples.")

# Modified main section that uses get_models instead of looking at config files
if __name__ == '__main__':
    args = parse_args()

    script_name = Path(__file__).stem
    logger = get_logger(script_name, True)
    notifier = PushoverNotifier()
    
    # Get models from directories with both name and size filters
    remaining_models = get_models(
        args.models_path, 
        model_name_filter=args.model_name,
        max_model_size=args.max_model_size,
        min_model_size=args.min_model_size
    )
        
    logger.info(f'Script started with models: {remaining_models}')
    
    # Rest of the code remains the same
    for model_name in remaining_models:

        logger.info(f"Running model: {model_name}")

        args.model_name = model_name
        script_config, model_config, llama_cpp_config = get_configs(args)
        runner = LogicProgramRunner(script_config, model_config, llama_cpp_config, logger)
        runner.run()
        logger.info(f"Finished Successfully: {model_name}")
        notifier.send(f"Finished Successfully: {model_name}", 'Info')
        
        remaining_models.remove(model_name)
            
    logger.info("All models completed successfully")
    notifier.send("All models completed successfully", 'Info')