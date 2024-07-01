# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from gcd_utils import GrammarConstrainedModel
from openai_utils import OpenAIModel

from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")

class SelfRefinementEngine:
    def __init__(self, args, current_round, refiner):
        self.args = args
        self.data_path = args.data_path
        # self.predicates_path = args.predicates_path
        self.load_dir = args.load_dir
        self.split = args.split
        self.sketcher_name = args.sketcher_name
        self.dataset_name = args.dataset_name
        self.current_round = current_round
        self.prompt_mode = args.prompt_mode
        self.openai_api = OpenAIModel(api_key, args.sketcher_name,
                                    #   args.stop_words, args.max_new_tokens
                                      )
        self.refiner = refiner if refiner else None
        
        self.refiner_name = args.refiner_path.split('/')[-1].split('.')[0] if refiner else self.sketcher_name
        

        self.logic_programs = self.load_logic_programs()
        self.ground_truth = self.load_ground_truth()
        self.predicates = self.load_predicates()

        self.parsing_error_prompt = {'FOLIO': self.parsing_prompt_folio,
                            'LogicNLI': self.parsing_prompt_folio}
        
        
        self.execution_error_prompt = {'FOLIO': self.execution_prompt_folio,
                            'LogicNLI': self.execution_prompt_folio}
            

        program_executor_map = {'FOLIO': FOL_Prover9_Program,
                                'LogicNLI': FOL_Prover9_Program}
        self.program_executor = program_executor_map[self.dataset_name]
                
        self.load_prompt_templates()
        
    def load_logic_programs(self):
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'            
            programs_path = os.path.join(self.load_dir, 'logic_programs', self.refiner_name, f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
            
        else:
            programs_path = os.path.join(self.load_dir, 'logic_programs', f'{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
            
        with open(programs_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def load_predicates(self):
        
        predicates_path = os.path.join(self.load_dir, 'logic_predicates', f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
        
        with open(predicates_path) as f:
            predicates = json.load(f)
        return predicates

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
            
            
    def parsing_prompt_folio(self, nl_statement, sketch, predicates):
        
        full_prompt = self.parsing_prompt_template.replace('[[NLSTATEMENT]]', nl_statement).replace('[[SKETCH]]', sketch).replace('[[PREDICATES]]', '\n'.join(predicates))
        
        grammar_predicates = [predicate.split(':::')[0].split('(')[0].strip() for predicate in predicates]
        
        grammar_predicates = [f'\"{predicate}\"' for predicate in grammar_predicates]
        
        grammar = self.grammar_template.replace('[[PREDICATES]]', ' | '.join(grammar_predicates))
        
        # print(full_prompt)
        # print('-'*50)
        # print(grammar)
        
        # sys.exit(0)
        
        return full_prompt, grammar
    
    
    def execution_prompt_folio(self, program, status):
        
        program_string = json.dumps(program).replace('{', '\\{').replace('}', '\\}')
        
        full_prompt = self.execution_prompt_template.replace('[[PROGRAM]]', program_string).replace('[[ERROR MESSAGE]]', status)
        
        return full_prompt
    
    def safe_execute_program(self, logic_program):
        program = self.program_executor(logic_program, self.dataset_name, self.prompt_mode)
        # cannot parse the program
        if program.flag == False:
            return 'parsing error', program.formula_error, program.nl_error
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            return 'execution error', error_message, None
        # successfully executed
        return 'success', '', None
    
    def single_round_self_refinement(self):
        outputs = []
        for example in tqdm(self.logic_programs):
            logic_program = example['raw_logic_programs']
            status, error, nl_error = self.safe_execute_program(logic_program)
        
            if status == 'parsing error':

                try:
                    
                    # open('this')
                    predicates = self.predicates[str(example['id'])]['logic_predicates']

                    full_prompt, grammar = self.parsing_error_prompt[self.dataset_name](nl_error, error, predicates)

                    if self.refiner:
                        response = self.refiner.invoke(full_prompt, self.task_description_parsing, grammar)
                    else:
                        response = self.openai_api.generate(full_prompt, self.task_description_parsing, {"type": "text"}).strip()

                    # print(response)

                    revised_program_string = json.dumps(logic_program, ensure_ascii=False).replace(error, response)

                    revised_program = json.loads(revised_program_string)

                except Exception as e:
                    print(f'Exception for {example["id"]}: {e}')
                    revised_program = logic_program


                # programs = [revised_program]
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': revised_program}
                outputs.append(output)
                
            # if status != 'success':
            elif status == 'execution error':
            # if not error_message == 'No Output': # this is not execution error, but parsing error
                # perform self-correction based on the error message
                full_prompt = self.execution_error_prompt[self.dataset_name](logic_program, error)

                try:

                    # open('this')

                    response_string = self.openai_api.generate(full_prompt, self.task_description_execution, {"type": "json_object"})

                    response = json.loads(response_string)

                    revised_program = response['Correct Program']

                    # programs = revised_program
                except Exception as e:
                    print(f'Exception for {example["id"]}: {e}')
                    revised_program = logic_program

                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': revised_program}
                outputs.append(output)
            else:
                outputs.append(example)
        # save results
        if not os.path.exists(os.path.join(self.load_dir, 'logic_programs', self.refiner_name)):
            os.makedirs(os.path.join(self.load_dir, 'logic_programs', self.refiner_name))

        # save outputs
        save_path = os.path.join(self.load_dir, 'logic_programs', self.refiner_name, f'self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
        # save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json'
        with open(save_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default='./outputs_1/')
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument('--sketcher_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--refiner_path', type=str)
    parser.add_argument('--n_gpu_layers', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    starting_round = args.self_refine_round + 1
    
    
    if args.refiner_path:
        
        # print(f"Using refiner model from {args.refiner_path}.")
        
        refiner=GrammarConstrainedModel(
            refiner_path=args.refiner_path,
            n_gpu_layers=args.n_gpu_layers,
        )
        
        args.refiner_name = args.refiner_path.split('/')[-1].split('.')[0]
        
    else:
        
        # print(f"Using OpenAI model {args.sketcher_name} as refiner.")
        refiner=None
        args.refiner_name = args.sketcher_name
    
    for round in range(starting_round, args.maximum_rounds+1):
        print(f"Round {round} self-refinement")
        engine = SelfRefinementEngine(args, round, refiner = refiner)
        engine.single_round_self_refinement()