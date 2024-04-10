# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel
import sys
import argparse

task_description = ("Given a set of Natural Language Premises and a Natural Language Question. "
"The task is extract the First-Order-Logic Predicates, and parse the Premises and Question into First-Order-Logic formulas.\n"
"The grammar of the first-order logic formular is defined as follows:\n"
"1) logical conjunction of expr1 and expr2: expr1 ∧ expr2\n"
"2) logical disjunction of expr1 and expr2: expr1 ∨ expr2\n"
"3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2\n"
"4) logical negation of expr1: ¬expr1\n"
"5) expr1 implies expr2: expr1 → expr2\n"
"6) expr1 if and only if expr2: expr1 ↔ expr2\n"
"7) logical universal quantification: ∀x\n"
"8) logical existential quantification: ∃x\n------\n")



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
        
        if self.prompt_mode == 'static':
            self.prompt_creator = {'FOLIO': self.static_prompt_folio,
                                'FOLIOv2': self.static_prompt_folio,
                                'ProntoQA': self.prompt_prontoqa,
                                'ProofWriter': self.prompt_proofwriter,
                                'LogicalDeduction': self.prompt_logicaldeduction, 
                                'AR-LSAT': self.prompt_arlsat}
            self.batch_logic_program_generation = self.static_batch_logic_program_generation
            
        elif self.prompt_mode == 'dynamic':
            self.prompt_creator = {'FOLIO': self.dynamic_prompt_folio,
                                'FOLIOv2': self.dynamic_prompt_folio,
                                'ProntoQA': self.prompt_prontoqa,
                                'ProofWriter': self.prompt_proofwriter,
                                'LogicalDeduction': self.prompt_logicaldeduction, 
                                'AR-LSAT': self.prompt_arlsat}
            self.batch_logic_program_generation = self.dynamic_batch_logic_program_generation
            
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        if self.dataset_name == 'AR-LSAT' and self.model_name == 'gpt-4':
            prompt_file = f'./models/prompts/{self.dataset_name}-long.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()

    def static_prompt_folio(self, test_data):
        problem = ' '.join(test_data['context'])
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def dynamic_prompt_folio(self, test_data, train_data):
            
        prompt = task_description

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

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
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
    
    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def load_dynamic_examples(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}_examples.json')) as f:
            dynamic_examples = json.load(f)
        return dynamic_examples

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                output = self.openai_api.generate(full_prompt)
                # print(full_prompt)
                programs = [output]

                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
            except Exception as e:
                print('Error in generating logic programs for example: ', example['id'], e)

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
            
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
                batch_outputs = self.openai_api.batch_generate(full_prompts)
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
                        output = self.openai_api.generate(full_prompt)
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
                batch_outputs = self.openai_api.batch_generate(full_prompts)
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
                        output = self.openai_api.generate(full_prompt)
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