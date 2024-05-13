# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel
import sys
import argparse

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
                        
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                            'FOLIOv2': self.prompt_folio}
            
        self.load_prompt_templates()
            
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        task_description_file = f'./models/task_descriptions/{self.dataset_name}.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
            
        with open(task_description_file, 'r') as f:
            self.task_description = f.read()

    def prompt_folio(self, test_data):
        problem = '\n'.join(test_data['context'])
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
                    
        return full_prompt    

class LogicProgramGenerator(PromptGenerator):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
  
    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''     
    def batch_logic_program_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)

        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
                         
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts, self.task_description)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    programs = [output]
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            # 'options': sample['options'],
                            'raw_logic_programs': programs}
                    outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt, self.task_description)
                        programs = [output]
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                # 'options': sample['options'],
                                'raw_logic_programs': programs}
                        outputs.append(output)
                    except KeyboardInterrupt:
                        sys.exit()
                    # except:
                    #     print('Error in generating logic programs for example: ', sample['id'])

        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


class Cheater:

    def __init__(self, args):
            
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
    
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def cheat(self):
        
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        outputs = []
        # split dataset into chunks
    
        for sample in raw_dataset:
            
            premises = '\n'.join(sample['context_fol'])
            programs = "First-Order-Logic Premises:\n\"\"\"\n" + premises
            
            programs += '\n\"\"\"'
            
            if 'question_fol' in sample.keys():
                question = sample['question_fol']
                programs += "\nFirst-Order-Logic Question:\n\"\"\"\n" + question + "\n\"\"\""
        
            output = {'id': sample['id'], 
                    # 'context': sample['context'],
                    'question': sample['question'], 
                    'answer': sample['answer'],
                    # 'options': sample['options'],
                    'raw_logic_programs': [programs]}
            outputs.append(output)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
                    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--cheat', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    if args.cheat:
        cheater = Cheater(args)
        cheater.cheat()
    else:
        logic_program_generator = LogicProgramGenerator(args)
        logic_program_generator.batch_logic_program_generation()