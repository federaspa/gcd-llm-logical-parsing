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
        self.prompt_mode = args.prompt_mode
        self.response_mode = args.response_mode
        
        if self.response_mode == 'json':
            self.response_format = { "type": "json_object" }
        
            if self.prompt_mode == 'static':
                self.prompt_creator = {'FOLIO': self.static_prompt_folio_json,
                                    'FOLIOv2': self.static_prompt_folio_json}
                
            elif self.prompt_mode == 'dynamic':
                self.prompt_creator = {'FOLIO': self.dynamic_prompt_folio_json,
                                    'FOLIOv2': self.dynamic_prompt_folio_json}
                
                
        elif self.response_mode == 'text':
            self.response_format = { "type": "text" }
        
            if self.prompt_mode == 'static':
                self.prompt_creator = {'FOLIO': self.static_prompt_folio_text,
                                    'FOLIOv2': self.static_prompt_folio_text}
                
            elif self.prompt_mode == 'dynamic':
                self.prompt_creator = {'FOLIO': self.dynamic_prompt_folio_text,
                                    'FOLIOv2': self.dynamic_prompt_folio_text}
            
        self.load_prompt_templates()
            
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}_{self.response_mode}.txt'
        task_description_file = f'./models/task_descriptions/{self.dataset_name}_{self.response_mode}.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
            
        with open(task_description_file, 'r') as f:
            self.task_description = f.read()

    def static_prompt_folio_text(self, test_data):
        problem = ' '.join(test_data['context'])
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    

    def static_prompt_folio_json(self, test_data):
        
        context = [f"\"{c}\"" for c in test_data['context']]
        problem = "[" + ','.join(context) + "]"
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def dynamic_prompt_folio_text(self, test_data, train_data):
        
        prompt = ""

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
    
    def dynamic_prompt_folio_json(self, test_data, train_data):
        
        prompt = ""
        
        if train_data:
            for story in train_data:   
                
                prompt += "{\\\"Natural Language Premises\\\":["
                for premise in story['context'][:-1]:
                    prompt += f"\\\"{premise}\\\","
                
                prompt += f"\\\"{story['context'][-1]}\\\"],"
                
                prompt += '\\\"Natural Language Question\\\":'
                prompt += "\\\"" + story['question'] + "\\\"}"
                prompt += "\n###\n" 
                
                
                prompt += "{\\\"First-Order-Logic Predicates\\\":["
                if 'predicates_fol' in story.keys():
                    for pred in story['predicates_fol'][:-1]:
                        
                        prompt += f"\\\"{pred}\\\","
                        
                prompt += f"\\\"{story['predicates_fol'][-1]}\\\"],"
                
                prompt += "\\\"First-Order-Logic Premises\\\":["
                for nl, fol in list(zip(story['context'], story['context_fol']))[:-1]:
                    
                    prompt += "\\\"" + fol + " ::: " + nl + "\\\","
                    
                prompt += "\\\"" + story['context_fol'][-1] + " ::: " + story['context'][-1] + "\\\"],"
                    
                prompt += "\\\"First-Order-Logic Question\\\":"
                prompt += "\\\"" + story['question_fol'] + "\\\"}"
                prompt += '\n------\n'
            
            
        prompt += "{\\\"Natural Language Premises\\\":["
        for premise in test_data['context'][:-1]:
            prompt += f"\\\"{premise}\\\","
        
        prompt += f"\\\"{test_data['context'][-1]}\\\"],"
        
        prompt += '\\\"Natural Language Question\\\":'
        prompt += "\\\"" + test_data['question'] + "\\\"}"
        prompt += "\n###\n" 
        
        return prompt
    
    def load_dynamic_examples(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}_examples.json')) as f:
            dynamic_examples = json.load(f)
        return dynamic_examples
    

class LogicProgramGenerator(PromptGenerator):
    def __init__(self, args):
        
        super().__init__(args)
        
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.prompt_mode = args.prompt_mode
        self.response_mode = args.response_mode

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        
        if self.response_mode == 'json':
            self.response_format = { "type": "json_object" }
        else:
            self.response_format = { "type": "text" }

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
        dynamic_examples = self.load_dynamic_examples(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            
            if self.prompt_mode == 'static':
                full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            
            elif self.prompt_mode == 'dynamic':
                full_prompts = [self.prompt_creator[self.dataset_name](example, dynamic_examples[str(example['id'])]) for example in chunk]
                        
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts, self.task_description, self.response_format)
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
                        output = self.openai_api.generate(full_prompt, self.task_description, self.response_format)
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

        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str)
    parser.add_argument('--response_mode', type=str, choices=['text', 'json'], default='text')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation()