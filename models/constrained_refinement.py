# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from backup_answer_generation import Backup_Answer_Generator
from utils import GrammarConstrainedModel
from utils import OpenAIModel

class SelfRefinementEngine:
    def __init__(self, args, current_round):
        self.args = args
        self.data_path = args.data_path
        self.predicates_path = args.predicates_path
        self.split = args.split
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.current_round = current_round
        self.prompt_mode = args.prompt_mode
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.constrained_model = GrammarConstrainedModel()
        

        self.logic_programs = self.load_logic_programs()
        self.ground_truth = self.load_ground_truth()
        self.predicates = self.load_predicates()

        self.parsing_error_prompt = {'FOLIO': self.parsing_prompt_folio,
                            'FOLIOv2': self.parsing_prompt_folio}
        
        
        self.execution_error_prompt = {'FOLIO': self.execution_prompt_folio,
                            'FOLIOv2': self.execution_prompt_folio}
            

        program_executor_map = {'FOLIO': FOL_Prover9_Program,
                                'FOLIOv2': FOL_Prover9_Program}
        self.program_executor = program_executor_map[self.dataset_name]
                
        self.load_prompt_templates()
        
        # self.backup_result_path = os.path.join(self.backup_path, f'{self.backup_strategy}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        
        # self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.backup_result_path)

    def load_logic_programs(self):
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'
        with open(os.path.join('./outputs/logic_programs', f'{prefix}{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def load_predicates(self):
        with open(os.path.join(self.predicates_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json')) as f:
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
        
        return full_prompt, grammar
    
    
    def execution_prompt_folio(self, program, status):
        
        full_prompt = self.execution_prompt_template.replace('[[PROGRAM]]', program).replace('[[ERROR MESSAGE]]', status)
        
        return full_prompt
    
    def safe_execute_program(self, id, logic_program, debug = False):
        program = self.program_executor(logic_program, self.dataset_name, self.prompt_mode)
        # cannot parse the program
        if program.flag == False:
            return 'parsing error', program.formula_error, program.parsing_error_index
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
            logic_program = example['raw_logic_programs'][0].strip()
            status, error, error_index = self.safe_execute_program(example['id'], logic_program)

            if status == 'parsing error':
                
                nl_statement = self.ground_truth[example['id']]['context'][error_index]
                predicates = self.predicates[str(example['id'])]['logic_predicates']
                
                full_prompt, grammar = self.parsing_error_prompt[self.dataset_name](nl_statement, error, predicates)
                    
                try:           
                    response = self.constrained_model.invoke(full_prompt, self.task_description_parsing, grammar)
                
                    response = response['choices'][0]['message']['content'].strip()
                                    
                    revised_program = logic_program.replace(error, response)
                except:
                    revised_program = logic_program
                
                programs = [revised_program]
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
                
            elif status == 'execution error':
            # if not error_message == 'No Output': # this is not execution error, but parsing error
                # perform self-correction based on the error message
                full_prompt = self.execution_error_prompt[self.dataset_name](logic_program, error)
                revised_program = self.openai_api.generate(full_prompt, self.task_description_execution).strip()
                programs = [revised_program]
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        # 'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
            else:
                outputs.append(example)
        # save results
        if not os.path.exists('./outputs/logic_programs'):
            os.makedirs('./outputs/logic_programs')

        # save outputs
        save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}.json'
        with open(save_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--predicates_path', type=str, default='./outputs/logic_predicates')
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='static')
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'Direct', 'CoT'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./baselines/results')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    starting_round = args.self_refine_round + 1
    
    for round in range(starting_round, args.maximum_rounds+1):
        print(f"Round {round} self-refinement")
        engine = SelfRefinementEngine(args, round)
        engine.single_round_self_refinement()