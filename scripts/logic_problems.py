import json
import os
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
from datetime import datetime
import yaml
from pprint import pp

from utils.utils import OSModel, PromptGenerator, send_notification, get_logger, calculate_perplexity

import traceback

@dataclass
class ScriptConfig:
    sketcher_name: str
    dataset_name: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    timeout_seconds: int|None
    stop_time: str|None
    two_steps: bool = False
    force_unconstrained: bool = False
    force_constrained: bool = False
    debug: bool = False

class LogicProgramGenerator(PromptGenerator):
    def __init__(self, script_config: ScriptConfig, model_config: dict, llama_cpp_config: dict):
        
        self.script_config = script_config
        
        super().__init__(self.script_config)
        
        self.sketcher_api = OSModel(llama_cpp_config)
        self.model_config = model_config
        
        self.stop_time = None
        if self.script_config.stop_time:
            try:
                self.stop_time = datetime.strptime(self.script_config.stop_time, "%d-%m-%y:%H-%M-%S")
            except ValueError as e:
                raise ValueError(f"Invalid stop_time format. Use dd-mm-yy:hh-mm-ss. Error: {str(e)}")
        
        self._unconstrained_generator = timeout(seconds=self.script_config.timeout_seconds)(self._unconstrained_generator_base)
        self._json_wrapper = timeout(seconds=self.script_config.timeout_seconds)(self._json_wrapper_base)
        self._constrained_generator = timeout(seconds=self.script_config.timeout_seconds)(self._constrained_generator_base)
        self.grammar = self.templates['constrained_grammar']
        
    def _check_time_limit(self):
        """Check if we've reached the stop time"""
        if self.stop_time and datetime.now() >= self.stop_time:
            logger.info(f"Stop time {self.stop_time} reached. Saving progress and exiting...")
            return True
        return False
    
    def _load_raw_dataset(self) -> List[dict]:
        dataset_path = Path(self.script_config.data_path) / self.script_config.dataset_name / f'{self.script_config.split}.json'
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    
    def _extract_constructs(self, sample: dict) -> List[str]:
        """Extract constructs from natural language using LLM"""
        
        if self.script_config.two_steps:
            
            user = self.prompter.extract_constructs(sample)
            response = self.sketcher_api.invoke(
                prompt=user,
                raw_grammar=self.templates['json_grammar'],
                model_config=self.model_config
            )
            
            content = response['choices'][0]['text']
            constructs = json.loads(content)
        
        else:
            constructs = {}
        
        self.grammar = self.prompter.get_grammar(constructs)

    def _unconstrained_generator_base(self, sample: dict) -> Tuple[str, float]:
        user = self.prompter.unconstrained(sample=sample)
        response = self.sketcher_api.invoke(
            prompt=user,
            # model_config=self.model_config
            model_config={}
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        return content, perplexity

    def _json_wrapper_base(self, unconstrained: str) -> dict:
        user = self.prompter.json_wrap(unconstrained)
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.templates['json_grammar'],
            model_config=self.model_config
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        return json.loads(content), perplexity
    
    def _constrained_generator_base(self, sample: str) -> dict:
        self._extract_constructs(sample)
        user = self.prompter.constrained(sample)
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.grammar,
            model_config=self.model_config
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
        save_path = Path(self.script_config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_file = save_path / f'{self.script_config.dataset_name}_{self.script_config.split}_{self.script_config.sketcher_name}.json'
        
        # raw_dataset, outputs, existing_samples, existing_ids = self._skip_existing(save_file=save_file, raw_dataset=raw_dataset)
        raw_dataset, outputs, existing_ids = self._skip_existing(save_file=save_file, raw_dataset=raw_dataset)
        
        # raw_dataset = [s for s in raw_dataset if s['id']>112]
        
        logger.info(f"Loaded {len(raw_dataset)} examples from {self.script_config.split} split.")

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
                if not 'logic_problem' in sample.keys() or self.script_config.force_unconstrained:
                    pbar.set_description_str(f"Generating unconstrained problem {sample['id']}")
                    unconstrained, perplexity = self._unconstrained_generator(nl_problem)
                    
                    pbar.set_description_str(f"Json wrapping problem {sample['id']}")
                    logic_problem, json_perplexity = self._json_wrapper(unconstrained)
                    
                    logic_problem['perplexity'] = (perplexity, json_perplexity)
                    
                    # if i % 20 == 0:
                    #     logger.debug(unconstrained)
                        
                else:
                    logic_problem = sample['logic_problem']
                    
                    
                # if i % 20 == 0:
                #     logger.debug(logic_problem)
                        
            
            except json.decoder.JSONDecodeError:
                print()
                logger.error(f"Unconstrained: Json wrapping error for sample {sample['id']}")
                logger.debug(unconstrained)
            
            except TimeoutError:
                print()
                logger.error(f"Unconstrained: Timeout error for sample {sample['id']}")

            except Exception as e:
                print()
                logger.error(f"Unconstrained: An error occurred for sample {sample['id']}: {str(e)}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                
            try:
                if not 'logic_problem_gcd' in sample.keys() or self.script_config.force_constrained:
                    pbar.set_description_str(f"Generating constrained problem {sample['id']}")
                    logic_problem_gcd, gcd_perplexity= self._constrained_generator(nl_problem)
                    
                    logic_problem_gcd['perplexity'] = gcd_perplexity
                else:
                    logic_problem_gcd = sample['logic_problem_gcd']
                
                # if i % 20 == 0:
                #     logger.debug(logic_problem_gcd)
                    
            except TimeoutError:
                print()
                logger.error(f"Constrained: Timeout occurred during constrained generation for sample {sample['id']}")
                
            except Exception as e:
                print()
                logger.error(f"Constrained: An error occurred for sample {sample['id']}: {str(e)}")
                logger.debug(f"Traceback:\n{traceback.format_exc()}")
                
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

def parse_args() -> ScriptConfig:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--two-steps', action='store_true', help='Extract predicates in first step')
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--timeout-seconds', type=int, default=None, help='Timeout in seconds for generation operations')
    parser.add_argument('--force-unconstrained', action='store_true')
    parser.add_argument('--force-constrained', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    return ScriptConfig(**vars(args))


def get_configs(script_config: ScriptConfig):
    # Load model-specific configuration
    sketcher_config_path = Path('configs/models') / f"{script_config.sketcher_name}.yml"
    with open(sketcher_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # Load general LlamaCpp generation configuration
    llamacpp_config_path = Path('configs') / 'llamacpp.yml'
    with open(llamacpp_config_path, 'r') as f:
        llama_cpp_config = yaml.safe_load(f)
    
    # Add model path to model config
    llama_cpp_config['model_path'] = str(Path(script_config.models_path) / f"{script_config.sketcher_name}.gguf")
    
    # Verify model path exists
    assert Path(llama_cpp_config['model_path']).exists()
    
    return script_config, model_config, llama_cpp_config

if __name__ == '__main__':
    args = parse_args()
    
    script_name = Path(__file__).stem
    logger, log_file_name = get_logger(script_name, args.debug)
    
    script_config, model_config, llama_cpp_config = get_configs(args)
    
    # for key, value in vars(script_config).items():
    #     logger.info(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    # for key, value in model_config.items():
    #     logger.info(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    # for key, value in llama_cpp_config.items():
    #     logger.info(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    # pp(script_config)
    # pp(model_config)
    # pp(llama_cpp_config)
    
    # sys.exit(0)
    
    try:
        generator = LogicProgramGenerator(
            script_config=script_config,
            model_config=model_config,
            llama_cpp_config=llama_cpp_config
        )
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