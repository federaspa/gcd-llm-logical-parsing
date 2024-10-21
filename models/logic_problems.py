# generate facts and rules based on the problem description

import json
import os
from tqdm.autonotebook import tqdm
from utils import OSModel, send_notification, get_logger
import argparse
from collections.abc import Callable
import traceback
import sys

class PromptGenerator:
    def __init__(self, args):
        self.dataset_name = args.dataset_name

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
            
        self.load_prompt_templates()
  
    def load_prompt_templates(self):

        generation_user = f'./prompts/conversion/user/{self.type}_conversion.txt'
        reasoning_user = f'./prompts/conversion/user/{self.type}_reasoning.txt'
        
        generation_system = f'./prompts/conversion/system/{self.type}_conversion.txt'
        reasoning_system = f'./prompts/conversion/system/{self.type}_reasoning.txt'
        
        with open(generation_user, 'r') as f:
            self.generation_template = f.read()
            
        with open(generation_system, 'r') as f:
            self.generation_system = f.read()
            
        with open(reasoning_user, 'r') as f:
            self.reasoning_template = f.read()
            
        with open(reasoning_system, 'r') as f:
            self.reasoning_system = f.read()
    
    def prompt_fol(self, sample:dict, mode:str, reasoning:str|None = None):
        
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'
        
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        
        if mode == 'reasoning':
            return self.reasoning_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
            
        elif mode == 'generation':
            return self.generation_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question).replace('[[reasoning]]', reasoning)
        
    # def prompt_reasoning_fol(self, sample):
    #     problem = '\n'.join(sample['context'])
    #     question = sample['question'].strip()
        
    #     full_prompt = self.reasoning_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
        
    #     return full_prompt

class LogicProgramGenerator(PromptGenerator):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sketcher_path = args.sketcher_path
        self.save_path = args.save_path
        self.n_gpu_layers = args.n_gpu_layers
        
        self.sketcher_api = OSModel(model_path=self.sketcher_path, n_gpu_layers=self.n_gpu_layers, verbose=args.verbose)
        self.sketcher_name = os.path.splitext(self.sketcher_path)[0].split('/')[-1]
        
        self.prompter:Callable = self.prompters[self.type]
        # self.reasoning_prompter:Callable[[dict],str] = self.reasoning_prompters[self.type]
            
    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def reasoning_generator(self, sample:dict) -> str:
        user = self.prompter(sample = sample, mode = 'reasoning')

        response = self.sketcher_api.invoke(
            user=user,
            task_description=self.reasoning_system,
        )
        
        content = response['choices'][0]['message']['content']
        
        return content
  
    def logic_program_generator(self, sample:dict, reasoning:str) -> dict:
        user = self.prompter(sample = sample, reasoning = reasoning, mode = 'generation')
        
        response = self.sketcher_api.invoke(
            user=user,
            task_description=self.generation_system,
            json_format=True
        )
        
        content = response['choices'][0]['message']['content']
        problem:dict = json.loads(content)
        
        problem['reasoning'] = reasoning
        
        return problem
  
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
                
                reasoning = self.reasoning_generator(sample)
            
                logic_problem = self.logic_program_generator(sample, reasoning)
                
                if i%20 == 0:
                    logger.debug(reasoning)
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
    parser.add_argument('--sketcher-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
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
    logger.info(f"Sketcher: {args.sketcher_path}")
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