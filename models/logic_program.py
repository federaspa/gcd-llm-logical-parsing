# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel
import sys
import argparse



class LogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.prompt_mode = args.prompt_mode

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        
        if args.response_format == 'json':
            self.response_format = { "type": "json_object" }
        else:
            self.response_format = { "type": "text" }
        
        if self.prompt_mode == 'static':
            self.prompt_creator = {'FOLIO': self.static_prompt_folio,
                                'FOLIOv2': self.static_prompt_folio,
                                'ProntoQA': self.prompt_prontoqa,
                                'ProofWriter': self.prompt_proofwriter}
            self.batch_logic_program_generation = self.static_batch_logic_program_generation
            
        elif self.prompt_mode == 'dynamic':
            self.prompt_creator = {'FOLIO': self.dynamic_prompt_folio,
                                'FOLIOv2': self.dynamic_prompt_folio,
                                'ProntoQA': self.prompt_prontoqa,
                                'ProofWriter': self.prompt_proofwriter}
            self.batch_logic_program_generation = self.dynamic_batch_logic_program_generation
            
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        task_description_file = f'.models/task_descriptions/{self.dataset_name}.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
            
        with open(task_description_file, 'r') as f:
            self.task_description = f.read()

    def static_prompt_folio(self, test_data):
        problem = ' '.join(test_data['context'])
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def dynamic_prompt_folio(self, test_data, train_data):
            
        prompt = self.task_description

        if train_data:
            for story in train_data:
                
                prompt += 'Natural Language Premises:\n"""\n'
                prompt += '\n'.join(story['context'])
                prompt += '\n"""\n'
                
                prompt += 'Natural Language Question:\n"""\n'
                prompt += story['question']
                prompt += '\n"""\n###\n'   
                
                
                prompt += 'First-Order-Logic Predicates:\n"""\n'
                if 'predicates_fol' in story.keys():
                    for pred in story['predicates_fol']:
                        prompt += pred + '\n'
                prompt += '"""\n'
                
                prompt += 'First-Order-Logic Premises:\n"""\n'
                for nl, fol in zip(story['context'], story['context_fol']):
                    prompt += fol + ' ::: ' + nl + '\n'
                prompt += '"""\n'
                    
                prompt += 'First-Order-Logic Question:\n"""\n'
                prompt += story['question_fol']
                prompt += '\n"""\n'
                prompt += '------\n'
            
        prompt += 'Natural Language Premises:\n"""\n'
        prompt += '\n'.join(test_data['context'])
        prompt += '\n"""\n'
        
        prompt += 'Natural Language Question:\n"""\n'
        prompt += test_data['question']
        prompt += '\n"""\n###\n'   
            
        return prompt
    
    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def load_dynamic_examples(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}_examples.json')) as f:
            dynamic_examples = json.load(f)
        return dynamic_examples
            
    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def static_batch_logic_program_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts, self.response_format)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    programs = [output]
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'options': sample['options'],
                            'raw_logic_programs': programs}
                    outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt, self.response_format)
                        programs = [output]
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                'options': sample['options'],
                                'raw_logic_programs': programs}
                        outputs.append(output)
                    except KeyboardInterrupt:
                        sys.exit()
                    except:
                        print('Error in generating logic programs for example: ', sample['id'])
                        


        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
            
    def dynamic_batch_logic_program_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        dynamic_examples = self.load_dynamic_examples(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example, dynamic_examples[str(example['id'])]) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts, self.response_format)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    programs = [output]
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'options': sample['options'],
                            'raw_logic_programs': programs}
                    outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        output = self.openai_api.generate(full_prompt, self.response_format)
                        programs = [output]
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                'options': sample['options'],
                                'raw_logic_programs': programs}
                        outputs.append(output)
                    except KeyboardInterrupt:
                        sys.exit()
                    except:
                        print('Error in generating logic programs for example: ', sample['id'])

        # # remove examples with duplicate ids from the result
        # outputs = list({output['id']: output for output in outputs}.values())
            # print(full_prompts[0])
            # break
        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str)
    parser.add_argument('--response_format', type=str, choices=['text', 'json'], default='text')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation()