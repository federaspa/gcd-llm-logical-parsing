# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
import re

from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from models.os_utils import OSModel

import traceback
from dotenv import load_dotenv

import traceback

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.sketcher_path = args.sketcher_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.prompt_mode = args.prompt_mode
                        

        self.parsing_error_prompt = {'FOLIO': self.parsing_prompt_folio,
                            'LogicNLI': self.parsing_prompt_folio}
        
        
        self.execution_error_prompt = {'FOLIO': self.execution_prompt_folio,
                            'LogicNLI': self.execution_prompt_folio}
        self.load_prompt_templates()
        
    def load_prompt_templates(self):
        parsing_prompt_file = f'./models/prompts/self-correct-parsing-{self.dataset_name}.txt'
        execution_prompt_file = f'./models/prompts/self-correct-execution-{self.dataset_name}.txt'
        task_description_parsing_file = f'./models/task_descriptions/self-correct-parsing-{self.dataset_name}.txt'
        task_description_execution_file = f'./models/task_descriptions/self-correct-execution-{self.dataset_name}.txt'
        grammar_file = f'./models/grammars/{self.dataset_name}.gbnf'
        
        with open(parsing_prompt_file, 'r') as f:
            self.parsing_prompt_template = f.read()
            
        with open(execution_prompt_file, 'r') as f:
            self.execution_prompt_template = f.read()
            
        with open(task_description_parsing_file, 'r') as f:
            self.task_description_parsing = f.read()
            
        with open(task_description_execution_file, 'r') as f:
            self.task_description_execution = f.read()
            
        with open(grammar_file, 'r') as f:
            self.grammar_template = f.read()
    
    def parsing_prompt_folio(self, sketch):

        full_prompt = self.parsing_prompt_template.replace('[[SKETCH]]', sketch)

        if not self.gcd:
            return full_prompt, None

        return full_prompt, self.grammar_template
    
    
    def execution_prompt_folio(self, program, status):
        
        program_string = json.dumps(program).replace('{', '\\{').replace('}', '\\}')
        
        full_prompt = self.execution_prompt_template.replace('[[PROGRAM]]', program_string).replace('[[ERROR MESSAGE]]', status)
        
        return full_prompt
   
    def load_dynamic_examples(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}_examples.json')) as f:
            dynamic_examples = json.load(f)
        return dynamic_examples

class SelfRefinementEngine(PromptGenerator):
    def __init__(self, args, current_round, refiner):
        
        raise NotImplementedError("This code is not up to date.")
        
        self.args = args
        
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.load_dir = args.load_dir
        self.split = args.split
        self.prompt_mode = args.prompt_mode
        
        self.current_round = current_round
        
        self.sketcher_name = os.path.splitext(args.sketcher_path)[0].split('/')[-1]
           
        self.refiner_api = OSModel(
            model_path=args.refiner_path,
            n_gpu_layers=args.n_gpu_layers,
            verbose=True
        )
        self.refiner_name = args.refiner_path.split('/')[-1].split('.')[0] if refiner else self.sketcher_name
    
        self.gcd = args.gcd
        self.gcd_dir = 'gcd' if self.gcd else 'no_gcd'



        self.logic_programs = self.load_logic_programs()
        self.ground_truth = self.load_ground_truth()
            

        # program_executor_map = {'FOLIO': FOL_Prover9_Program,
        #                         'LogicNLI': FOL_Prover9_Program}
        # self.program_executor = program_executor_map[self.dataset_name]
        self.program_executor = FOL_Prover9_Program
                
        self.load_prompt_templates()
        
    def load_logic_programs(self):
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'            
            programs_path = os.path.join(self.load_dir, 'logic_programs', self.gcd_dir, self.refiner_name, f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
            
        else:
            programs_path = os.path.join(self.load_dir, 'logic_programs', f'{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
            
        with open(programs_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

    def load_ground_truth(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json'), 'r') as f:
            ground_truth_raw = json.load(f)
            
        ground_truth = {
            point['id']: {
                'context': point['context'],
                'context_fol': point['context_fol'],
                'question': point['question'],
            }
        for point in ground_truth_raw
        }
            
        for point in ground_truth_raw:
            if 'question_fol' in point:
                key = point['id']
                ground_truth[key].update({
                'question_fol': point['question_fol']
                })

        return ground_truth
    
    def safe_execute_program(self, logic_program):
        program = self.program_executor(logic_program)
        # cannot parse the program
        if program.flag == False:
            return 'parsing error', program.formula_error
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            return 'execution error', error_message
        # successfully executed
        return 'success', ''
    
    def single_round_self_refinement(self):
        
        save_path = os.path.join(self.load_dir, 'logic_programs', self.gcd_dir, self.refiner_name, f'self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
        
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping.")
            return
        
        outputs = []
        for example in tqdm(self.logic_programs):

            if 'skip' in example.keys():
                if example['skip']:
                    outputs.append(example)
                    # print(f'Skipped {example["id"]}')
                    continue

            logic_program = example['raw_logic_programs']
            status, error = self.safe_execute_program(logic_program)

            if status == 'parsing error' and error:

                try:
                    
                    # open('this')

                    full_prompt, grammar = self.parsing_error_prompt[self.dataset_name](error)

                except Exception as e:
                    print(f'Exception for {example["id"]} for parsing prompt generation: {e}')
                    print('This sample will be skipped in the next iterations.')
                    revised_program = logic_program
                    # programs = [revised_program]
                    output = {'id': example['id'],
                            'context': example['context'],
                            'question': example['question'],
                            'answer': example['answer'],
                            # 'options': example['options'],
                            'raw_logic_programs': revised_program,
                              'skip': True}
                    outputs.append(output)
                    continue

                try:
                        
                    response = self.refiner_api.invoke(full_prompt, self.task_description_parsing, grammar)
                        
                    response = re.sub(r'[\n\t]', '', response)

                    revised_program_string = json.dumps(logic_program, ensure_ascii=False).replace(error, response)

                    revised_program = json.loads(revised_program_string)


                except Exception as e:
                    print(f'Exception for {example["id"]} for parsing response generation: {traceback.format_exc()}')
                    revised_program = logic_program


                # programs = [revised_program]
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': revised_program,
                          'skip': False}
                outputs.append(output)
                
            # if status != 'success':
            elif status == 'execution error' and error:
            # if not error_message == 'No Output': # this is not execution error, but parsing error
                # perform self-correction based on the error message
                full_prompt = self.execution_error_prompt[self.dataset_name](logic_program, error)

                try:

                    # open('this')

                    response_string = self.refiner_api.generate(full_prompt, self.task_description_execution, {"type": "json_object"})

                    response = json.loads(response_string)

                    assert 'Correct Program' in response.keys(), 'Correct Program not in response.keys()'
                    revised_program = response['Correct Program']

                    # programs = revised_program
                except Exception as e:
                    print(f'Exception for {example["id"]} for execution response: {e}')
                    # print(traceback.format_exc())
                    revised_program = logic_program

                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': revised_program,
                          'skip': False}
                outputs.append(output)
            elif status == 'success':
                example.update({'skip':True})
                outputs.append(example)

            else:
                # print(f'Something went wrong with {example["id"]}. This sample will be skipped in the next iterations.')
                example.update({'skip':True})
                outputs.append(example)

        # save results
        if not os.path.exists(os.path.join(self.load_dir, 'logic_programs', self.gcd_dir, self.refiner_name)):
            os.makedirs(os.path.join(self.load_dir, 'logic_programs', self.gcd_dir, self.refiner_name))

        # save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json'
        with open(save_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--sketcher_path', type=str, required=True)
    parser.add_argument('--refiner_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='./outputs/')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    
    parser.add_argument('--starting_round', type=int, default=0)
    parser.add_argument('--maximum_rounds', type=int, default=3)
    
    parser.add_argument('--n_gpu_layers', type=int, default=0)
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    starting_round = args.starting_round + 1
    
    
        
    refiner=OSModel(
        model_path=args.refiner_path,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True
    )
    
    args.refiner_name = args.refiner_path.split('/')[-1].split('.')[0]
        
    for round in range(starting_round, args.maximum_rounds+1):
        print(f"Round {round} self-refinement")
        engine = SelfRefinementEngine(args, round, refiner = refiner)
        engine.single_round_self_refinement()