# generate facts and rules based on the problem description

import json
import os
from tqdm import tqdm
from openai_utils import OpenAIModel
import sys
import argparse

from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.sketcher_name = args.sketcher
        self.predicates_path = args.predicates_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.prompt_mode = args.prompt_mode
                        
        if self.prompt_mode == 'static':
            self.prompt_creator = {'FOLIO': self.static_prompt_folio,
                                'FOLIOv2': self.static_prompt_folio}
            
        elif self.prompt_mode == 'dynamic':
            self.prompt_creator = {'FOLIO': self.dynamic_prompt_folio,
                                'FOLIOv2': self.dynamic_prompt_folio}
            
        self.load_prompt_templates()
        
        if 'FOLIO' in self.dataset_name:
            self.predicates = self.load_predicates_folio()
  
    def load_prompt_templates(self):

        prompt_file = f'./models/prompts/{self.dataset_name}_{self.prompt_mode}.txt'
        task_description_file = f'./models/task_descriptions/{self.dataset_name}.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
            
        with open(task_description_file, 'r') as f:
            self.task_description = f.read()
            
    def load_predicates_folio(self):
        with open(os.path.join(self.predicates_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')) as f:
            predicates = json.load(f)
        return predicates
    
    def static_prompt_folio(self, test_data):
        problem = '\n'.join(test_data['context'])
        question = test_data['question'].strip()
        predicates = '\n'.join(self.predicates[str(test_data['id'])]['logic_predicates'])
        
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question).replace('[[PREDICATES]]', predicates)

        return full_prompt

    def dynamic_prompt_folio(self, test_data, train_data):
        
        prompt = self.prompt_template
        
        if train_data:
            for i,story in enumerate(train_data[:5]):
                                
                prompt = prompt.replace(f'[[NLPROBLEM{i+1}]]', '\n'.join(story['context']))
                
                prompt = prompt.replace(f'[[NLQUESTION{i+1}]]', story['question'])
                
                prompt = prompt.replace(f'[[PREDICATES{i+1}]]', '\n'.join(story['logic_predicates']))
        
                if 'question_fol' in story.keys():
                    fol_question = story['question_fol'] + ' ::: ' + story['question']

                    prompt = prompt.replace(f'[[FOLQUESTION{i+1}]]', fol_question)
                else:
                    prompt = prompt.replace(f',\n"First-Order-Logic Question": "[[FOLQUESTION{i+1}]]"', '')
                            
                fol_rules = ''
                for nl, fol in zip(story['context'], story['context_fol']):
                    fol_rules += fol + ' ::: ' + nl + '\\n'
                    
                prompt = prompt.replace(f'[[FOLRULES{i+1}]]', fol_rules)
        
        prompt = prompt.replace('[[PROBLEM]]', '\n'.join(test_data['context']))
        prompt = prompt.replace('[[QUESTION]]', test_data['question'])
        prompt = prompt.replace('[[PREDICATES]]', '\n'.join(self.predicates[str(test_data['id'])]['logic_predicates']))

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
        self.sketcher_name = args.sketcher
        self.save_path = args.save_path
        self.prompt_mode = args.prompt_mode

        self.openai_api = OpenAIModel(api_key, args.sketcher, 
                                    #   args.stop_words, args.max_new_tokens
                                      )
        
        
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
        if self.prompt_mode == 'dynamic':
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
                batch_outputs = self.openai_api.batch_generate(full_prompts, self.task_description)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    
                    # programs = [output]
                    
                    programs = json.loads(output)
                    
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'raw_logic_programs': programs}
                    
                    if 'FOLIO' in self.dataset_name:
                        output['predicates'] = self.predicates[str(sample['id'])]['logic_predicates']
                    
                    outputs.append(output)
                    
            except KeyboardInterrupt:
                sys.exit()
                
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):

                    output = self.openai_api.generate(full_prompt, self.task_description)
                    # programs = [output]
                    programs = json.loads(output)
                    
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'raw_logic_programs': programs}
                    
                    if 'FOLIO' in self.dataset_name:
                        output['predicates'] = self.predicates[str(sample['id'])]['logic_predicates']
                    
                    outputs.append(output)
                    # except:
                    #     print('Error in generating logic programs for example: ', sample['id'])

        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)


class Cheater:

    def __init__(self, args):
            
        raise NotImplementedError("This class is not implemented yet.")
            
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sketcher_name = args.sketcher
        self.save_path = args.save_path
        self.prompt_mode = args.prompt_mode
    
    
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
            
            premises_list = sample['context_fol']
            
            if 'question_fol' in sample.keys():
                question = sample['question_fol']
            else:
                question = premises_list.pop()
            
            premises = '\n'.join(premises_list)
            programs = "First-Order-Logic Rules:\n\"\"\"\n" + premises
            
            programs += '\n\"\"\"'
            

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
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
                    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--predicates_path', type=str, default='./outputs/logic_predicates')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--sketcher_name', type=str, default='gpt-3.5-turbo')
    # parser.add_argument('--stop_words', type=str, default='------')
    # parser.add_argument('--max_new_tokens', type=int, default=1024)
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