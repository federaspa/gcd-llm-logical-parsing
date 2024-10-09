# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from os_utils import OSModel
import sys
import argparse
from collections.abc import Callable, Awaitable

from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")

class PromptGenerator:
    def __init__(self, args):
        self.dataset_name = args.dataset_name

        self.generation_prompt_creator = {'FOLIO': self.prompt_generation_folio,
                            'FOLIOv2': self.prompt_generation_folio,
                                'LogicNLI': self.prompt_generation_folio
                                }
        
        self.reasoning_prompt_creator = {'FOLIO': self.prompt_reasoning_folio,
                            'FOLIOv2': self.prompt_reasoning_folio,
                                'LogicNLI': self.prompt_reasoning_folio
                                }
            
        self.load_prompt_templates()
  
    def load_prompt_templates(self):

        generation_user = f'./prompts/generation/user/{self.dataset_name}_program.txt'
        generation_system = f'./prompts/generation/system/{self.dataset_name}_program.txt'
        
        reasoning_user = f'./prompts/reasoning/user/{self.dataset_name}_program.txt'
        reasoning_system = f'./prompts/reasoning/system/{self.dataset_name}_program.txt'
        
        with open(generation_user, 'r') as f:
            self.generation_template = f.read()
            
        with open(generation_system, 'r') as f:
            self.generation_system = f.read()
            
        with open(reasoning_user, 'r') as f:
            self.reasoning_template = f.read()
            
        with open(reasoning_system, 'r') as f:
            self.reasoning_system = f.read()
    
    def prompt_generation_folio(self, sample:dict, reasoning:str):
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        
        full_prompt = self.generation_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question).replace('[[reasoning]]', reasoning)
        
        return full_prompt
    
    def prompt_reasoning_folio(self, sample):
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        
        full_prompt = self.reasoning_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
        
        return full_prompt

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
        
        self.generation_prompt_creator:Callable[[dict],str] = self.generation_prompt_creator[self.dataset_name]
        self.reasoning_prompt_creator:Callable[[dict],str] = self.reasoning_prompt_creator[self.dataset_name]
            
    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def reasoning_generator(self, sample:dict) -> str:
        user = self.reasoning_prompt_creator(sample)

        response = self.sketcher_api.invoke(
            user=user,
            task_description=self.reasoning_system,
        )
        
        content = response['choices'][0]['message']['content']
        
        
        return content
  
    def logic_program_generator(self, sample:dict, reasoning:str) -> dict:
        user = self.generation_prompt_creator(sample, reasoning)
        
        response = self.sketcher_api.invoke(
            user=user,
            task_description=self.generation_system,
            json_format=True
        )
        
        content = response['choices'][0]['message']['content']
        program:dict = json.loads(content)
        
        program['reasoning'] = reasoning
        
        return program
  
    def run(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset()

        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        outputs = []
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        for i, sample in enumerate(tqdm(raw_dataset)):
            
            
            reasoning = self.reasoning_generator(sample)
        
            fol_problem = self.logic_program_generator(sample, reasoning)
            
            if i%100 == 0:
                print(reasoning)
                print(fol_problem)
            
            output = {'id': sample['id'], 
                        'nl_problem': {
                    'nl_rules': sample['context'],
                    'nl_conc': sample['question']
                        },
                    'answer': sample['answer'],
                    'fol_problem': fol_problem}
            
            outputs.append(output)
            
            # save outputs
            with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json'), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(outputs)} examples.")
        

   
     
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/generate_fol')
    parser.add_argument('--n_gpu_layers', type=int, default=0)
    parser.add_argument('--sketcher_path', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.run()