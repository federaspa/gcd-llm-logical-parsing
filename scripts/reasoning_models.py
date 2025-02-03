from logic_problems import get_configs, parse_args, LogicProgramRunner, ScriptConfig, get_logger
from tqdm import tqdm

import json
import time
import sys
import os
import traceback

from datetime import datetime
from pathlib import Path


class BaselineGenerator(LogicProgramRunner):
    def __init__(self, script_config: ScriptConfig, model_config: dict, llama_cpp_config: dict, logger):
        super().__init__(script_config, model_config, llama_cpp_config, logger)
        
    def run(self):
        # Setup paths and load data
        raw_dataset = self._load_raw_dataset()
        
        raw_dataset = raw_dataset[self.config.starting_sample:]

        save_file = self._prepare_save_file()
        raw_dataset, outputs, existing_ids = self._skip_existing(save_file, raw_dataset)
        
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
                pbar.set_description_str(f"Generating unconstrained problem {sample['id']}/{len(raw_dataset)}")
                logic_problem = self._generate_problem(nl_problem, sample["id"], "unconstrained")
                output['logic_problem'] = logic_problem
            else:
                output['logic_problem'] = sample['logic_problem']

            output['logic_problem_json'] = None
            output['logic_problem_gcd'] = None

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
            self.logger.info("Script stopped due to reaching stop time")
        else:
            self.logger.info(f"Generated {len(outputs)} examples.")

if __name__ == '__main__':
    args = parse_args()

    script_name = Path(__file__).stem
    logger, log_file_name = get_logger(script_name, args.debug)
    
    if args.start_time:
        start_time = datetime.strptime(args.start_time, "%d-%m-%y:%H-%M-%S")
        
        if datetime.now() < start_time:
            
            logger.info(f"Waiting to start script at {start_time}. Current time: {datetime.now()}, {(start_time - datetime.now()).total_seconds()} remaining")
        
        while datetime.now() < start_time:
            time.sleep((start_time - datetime.now()).total_seconds())
            
        
    logger.info('Script started')
    
    try:
        script_config, model_config, llama_cpp_config = get_configs(args)
        runner = BaselineGenerator(script_config, model_config, llama_cpp_config, logger)
        runner.run()
        logger.info("Finished Successfully")
        sys.exit()
        
    # except ValueError as e:
    #     if "Failed to load model from file:":
    #         logger.warning(f'Failed to load model, retrying in {retry_timeout} seconds')
    #         time.sleep(retry_timeout)
        
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        os.remove(f"./{log_file_name}")
        sys.exit(0)
        
    except Exception as e:
        error_message = f"A fatal error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_message)
        sys.exit(0)
        