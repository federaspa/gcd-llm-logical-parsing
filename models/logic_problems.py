# generate facts and rules based on the problem description

import json
import os
from tqdm.autonotebook import tqdm
from utils import OSModel, send_notification, get_logger
import argparse
from collections.abc import Callable
import traceback
import sys
from typing import Tuple

class PromptGenerator:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.model_name = args.sketcher_name

        self.prompters = {'FOL': self.prompt_fol,
                            # 'FOLIOv2': self.prompt_generation_folio,
                            #     'LogicNLI': self.prompt_generation_folio
                                }
        self.types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'LogicNLI': 'FOL'
        }
        
        self.type = self.types[self.dataset_name]
            
        self.load_templates()
  
    def load_templates(self):

        structured_user = f'./prompts/conversion/{self.type}_structured.txt'
        unstructured_user = f'./prompts/conversion/{self.type}_unstructured.txt'
        
        prompt_template = 'prompts/prompt_templates/' 
        prompt_template += 'gemma.txt' if 'gemma' in self.model_name else 'llama.txt'
        
        with open(prompt_template) as f:
            prompt_template = f.read()
        
        with open(structured_user, 'r') as f:
            structured_user = f.read()
            self.structured_user = prompt_template.replace('[[user]]', structured_user)
            
            
        with open(unstructured_user, 'r') as f:
            unstructured_user = f.read()
            self.unstructured_user = prompt_template.replace('[[user]]', unstructured_user)
            
        with open('./LLMs/grammars/json.gbnf') as f:
            self.json_grammar = f.read()
    
    def prompt_fol(self, mode:str, sample:dict|None = None, unstructured:str|None = None):
        
        assert mode in ['structured', 'unstructured'], 'wrong or no prompting mode specified'
        
        if mode == 'unstructured':
        
            problem = '\n'.join(sample['context'])
            question = sample['question'].strip()
        
            return self.unstructured_user.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
            
        elif mode == 'structured':
            return self.structured_user.replace('[[unstructured]]', unstructured)
        
class LogicProgramGenerator(PromptGenerator):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.save_path = args.save_path
        self.n_gpu_layers = args.n_gpu_layers
        
        
        self.sketcher_path = os.path.join(args.models_path, args.sketcher_name)
        
        assert os.path.exists(self.sketcher_path)
        
        self.sketcher_api = OSModel(model_path=self.sketcher_path, n_gpu_layers=self.n_gpu_layers, verbose=args.verbose)
        self.sketcher_name = os.path.splitext(self.sketcher_path)[0].split('/')[-1]
        
        self.prompter:Callable = self.prompters[self.type]
        # self.reasoning_prompter:Callable[[dict],str] = self.reasoning_prompters[self.type]
        
    
    def _handle_logprobs(self, logprobs):
        logprobs['token_logprobs'] = [float(p) for p in logprobs['token_logprobs']]    
        for step in logprobs['top_logprobs']:
            for key, value in step.items():
                step[key] = float(value)
                
        return logprobs
        
    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def unstructured_generator(self, sample:dict) -> Tuple[str, dict]:
        user = self.prompter(sample = sample, mode = 'unstructured')

        response = self.sketcher_api.invoke(
            prompt=user,
        )
        
        content = response['choices'][0]['text']
        logprobs = self._handle_logprobs(response['choices'][0]['logprobs'])
        
        return content, logprobs
  
    def structured_generator(self, unstructured:str) -> Tuple[dict, dict]:
        user = self.prompter(mode = 'structured', unstructured = unstructured)
        
        response = self.sketcher_api.invoke(
            prompt=user,
            raw_grammar=self.json_grammar
        )
        
        content = response['choices'][0]['text']
        logprobs = self._handle_logprobs(response['choices'][0]['logprobs'])
        
        problem:dict = json.loads(content)
        
        return problem, logprobs
  
    def run(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()

        save_file = os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')

        outputs = []

        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                outputs = json.load(f)

            existing_ids = [s["id"] for s in outputs]
            raw_dataset = [s for s in raw_dataset if s["id"] not in existing_ids]

        logger.info(f"{len(outputs)} already exist.\nLoaded {len(raw_dataset)} examples from {self.split} split.")

        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        for i, sample in enumerate(tqdm(raw_dataset)):
            
            try:
                
                unstructured, unstructured_logprobs = self.unstructured_generator(sample)
            
                
                if i%20 == 0:
                    logger.debug(unstructured)
                    
                logic_problem, structured_logprobs = self.structured_generator(unstructured)
                
                logic_problem['unstructured_logprobs'] = unstructured_logprobs
                logic_problem['structured_logprobs'] = structured_logprobs
                    
                if i%20 == 0:
                    logger.debug(logic_problem)
                
                output = {'id': sample['id'], 
                            'nl_problem': {
                        'nl_rules': sample['context'],
                        'nl_conc': sample['question']
                            },
                        'answer': sample['answer'],
                        'logic_problem': logic_problem}
                
                outputs.append(output)
                
            except Exception as e:
                
                # Get the full error traceback
                error_message = f"An error occurred for sample {sample['id']}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                
                # Send error notification
                send_notification(error_message, "logic_problems.py sample error")
                
                # Optionally, you can also print the error message
                logger.error(error_message)
            
            # save outputs
            with open(save_file, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

        logger.info(f"Generated {len(outputs)} examples.")
        

   
     
def parse_args():
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
    return args

if __name__ == '__main__':
    
    # Get the name of the current script
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    logger = get_logger(script_name)

    args = parse_args()
    
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Sketcher: {args.sketcher_name}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Save path: {args.save_path}")
    
    logic_program_generator = LogicProgramGenerator(args)
    
    try:
        logic_program_generator.run()
        
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        sys.exit(0)
        
    except Exception as e:
        error_message = f"A fatal error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        send_notification(error_message, "logic_problems.py fatal error")
        logger.error(error_message)
        sys.exit(0)
        
    logger.info("Finished Successfully")
    send_notification("Yippiee!", "logic_problems.py finished successfully")