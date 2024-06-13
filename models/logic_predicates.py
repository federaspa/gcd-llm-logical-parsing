# generate facts and rules based on the problem description

import json
import os
import re
from tqdm import tqdm
from openai_utils import OpenAIModel
from datetime import datetime
import argparse

from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")

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
        prompt_file = f'./models/prompts/{self.dataset_name}_predicates.txt'
        task_description_file = f'./models/task_descriptions/{self.dataset_name}_predicates.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
            
        with open(task_description_file, 'r') as f:
            self.task_description = f.read()

    def prompt_folio(self, test_data):
        problem = '\n'.join(test_data['context'])
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
    
        return full_prompt    

class PredicatesGenerator(PromptGenerator):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sketcher_name = args.sketcher_name
        self.save_path = args.save_path

        self.openai_api = OpenAIModel(api_key, args.sketcher_name, args.dataset_name)
        
        
    def parse_predicates(self, string_predicates):
        
        
        raw_predicates = json.loads(string_predicates)
    
        predicates = raw_predicates["First-Order-Logic Predicates"].split('\n')
                
        predicates = [predicate.strip() for predicate in predicates if re.sub(r'(?:[\',\",\`,\-,\s*])','', predicate)]
            
        return predicates
    
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
            
        full_prompts = {example['id']: self.prompt_creator[self.dataset_name](example) for example in raw_dataset}
                
            
        print(f"Sending batch job to OpenAI API.")
        batch = self.openai_api.batch_generate(full_prompts, self.task_description)
        
        print('Job submitted with id: ', batch.id)
        
        if not os.path.exists('active_requests'):
            os.makedirs('active_requests')
            
        active_requests_path = os.path.join('active_requests', f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        if os.path.exists(active_requests_path):
            active_requests = json.load(open(active_requests_path))
                
            active_requests[f'batch_{batch.created_at}'] = batch.id
            
        else:
            active_requests = {f'batch_{batch.created_at}': batch.id}
            
        with open((active_requests_path), 'w') as f:
            json.dump(active_requests, f, indent=2, ensure_ascii=False)
            
    # def batch_logic_program_generation(self, batch_size = 10):
    #     # load raw dataset
    #     raw_dataset = self.load_raw_dataset(self.split)

    #     print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
    #     outputs = {}
    #     # split dataset into chunks
    #     dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
    #     for chunk in tqdm(dataset_chunks):
            
    #         full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
                         
    #         try:
    #             batch_outputs = self.openai_api.batch_generate(full_prompts, self.task_description)
    #             # create output
    #             for sample, output in zip(chunk, batch_outputs):
                    
    #                 try:
    #                     logic_predicates= self.parse_predicates(output)
                        
    #                 except:
    #                     logic_predicates = output
                    
    #                 outputs[sample['id']] = { 
    #                             'context': sample['context'],
    #                             'question': sample['question'], 
    #                             'logic_predicates': logic_predicates}
                    
    #         except KeyboardInterrupt:
    #             sys.exit()
                        
    #         except:
    #             # generate one by one if batch generation fails
    #             for sample, full_prompt in zip(chunk, full_prompts):

    #                 output = self.openai_api.generate(full_prompt, self.task_description)
                    
    #                 try:
    #                     logic_predicates= self.parse_predicates(output)
                        
    #                 except:
    #                     logic_predicates = output
                
    #                 outputs[sample['id']] = { 
    #                         'context': sample['context'],
    #                         'question': sample['question'], 
    #                         'logic_predicates': logic_predicates}

    #                 # except:
    #                 #     print('Error in generating logic programs for example: ', sample['id'])

    #     print(f"Generated {len(outputs)} examples.")
        
    #     # save outputs
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
        
    #     with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json'), 'w') as f:
    #         json.dump(outputs, f, indent=2, ensure_ascii=False)

                    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_predicates')
    parser.add_argument('--sketcher_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    logic_program_generator = PredicatesGenerator(args)
    logic_program_generator.batch_logic_program_generation()