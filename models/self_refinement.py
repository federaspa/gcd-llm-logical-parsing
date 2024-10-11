# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
import sys

from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from utils import OSModel, get_logger, send_notification
from typing import Tuple, List, Callable

import traceback

script_name = os.path.splitext(os.path.basename(__file__))[0]

logger = get_logger(script_name)

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.sketcher_path = args.sketcher_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.prompt_mode = args.prompt_mode
                        

        self.parsing_error_prompters: Callable = {'FOLIO': self.parsing_prompt_folio,
                            'LogicNLI': self.parsing_prompt_folio}
        
        
        self.execution_error_prompters: Callable = {'FOLIO': self.execution_prompt_folio,
                            'LogicNLI': self.execution_prompt_folio}
        self.load_prompt_templates()
        
    def load_prompt_templates(self):
        parsing_prompt_file = f'./prompts/correction/user/parsing-{self.dataset_name}.txt'
        execution_prompt_file = f'./prompts/correction/user/execution-{self.dataset_name}.txt'
        task_description_parsing_file = f'./prompts/correction/system/parsing-{self.dataset_name}.txt'
        task_description_execution_file = f'./prompts/correction/system/execution-{self.dataset_name}.txt'
        grammar_file = f'./LLMs/grammars/{self.dataset_name}.gbnf'
        
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
    
    def parsing_prompt_folio(self, logic_problem, error):

        problem = '\n'.join(f for f in logic_problem['fol_rules'])
        problem += '\n' + logic_problem['fol_conc']
        
        full_prompt = self.parsing_prompt_template.replace('[[PROBLEM]]', problem).replace('[[ERROR]]', error)

        if not self.gcd:
            return full_prompt, None
        
        predicates = '\"' + logic_problem['fol_preds'][0].split('(')[0] + '\"'
        for pred in logic_problem['fol_preds'][1:]:
            predicates += ' | \"' + pred.split('(')[0] + '\"'
            
        grammar = self.grammar_template.replace('[[PREDICATES]]', predicates)
        

        return full_prompt, grammar
    
    
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
        
        raise NotImplemented('missing execution fixing')
        
        #super init
        super().__init__(args)
        
        self.args = args
        
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.load_dir = args.load_dir
        self.split = args.split
        self.prompt_mode = args.prompt_mode
        
        self.current_round = current_round
        
        self.sketcher_name = os.path.splitext(args.sketcher_path)[0].split('/')[-1]
           
        self.refiner_api = refiner
        self.refiner_name = os.path.splitext(args.refiner_path)[0].split('/')[-1]
        
        self.gcd = args.gcd
        self.gcd_dir = 'GCD' if self.gcd else 'NO_GCD'


        self.parsing_error_prompter = self.parsing_error_prompters[self.dataset_name]
        self.execution_error_prompter = self.execution_error_prompters[self.dataset_name]
        
        self.program_executor = FOL_Prover9_Program
                
        self.load_prompt_templates()
        
    def load_logic_problems(self) -> List[dict]:
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'            
            programs_path = os.path.join(self.load_dir, 'logic_programs', self.gcd_dir, self.refiner_name, f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        else:
            programs_path = os.path.join(self.load_dir, 'logic_programs', f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        with open(programs_path) as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    
    def parsing_correction_generator(self, logic_problem:dict, error:str) -> dict:
        
        user, grammar = self.parsing_error_prompter(logic_problem, error)
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.task_description_parsing,
            raw_grammar=grammar
        )
        
        content = response['choices'][0]['message']['content']
        
        logger.debug(user + '\n\n' + content)
        
        return content
    
    def execution_correction_generation(self, error:str) -> dict:
        
        user, grammar = self.execution_error_prompter(error)
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.task_description_parsing,
            raw_grammar=grammar
        )
        
        content = response['choices'][0]['message']['content']
        
        
        return content
  
    
    def safe_execute_program(self, logic_program:dict) -> Tuple[str,str, str]:
        program = self.program_executor(logic_program)
        # cannot parse the program
        if program.flag == False:
            return 'N/A', 'parsing error', program.formula_error
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            return 'N/A', 'execution error', error_message
        # successfully executed
        return answer, 'success', ''
    
    def single_round_self_refinement(self):
        
        logic_problems = self.load_logic_problems()
        
        save_path = os.path.join(self.load_dir, 'refinement', self.gcd_dir, self.refiner_name)
        save_file = os.path.join(save_path, f'self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        outputs = []

        if os.path.exists(save_file):
            with open(save_file, 'r') as f:
                outputs = json.load(f)

            existing_ids = [s["id"] for s in outputs]
            logic_problems = [s for s in logic_problems if s["id"] not in existing_ids]

        logger.info(f"{len(outputs)} already exist.\nLoaded {len(logic_problems)} examples from {self.split} split.")
        
        for sample in tqdm(logic_problems):

            if 'skip' in sample.keys():
                if sample['skip']:
                    outputs.append(sample)
                    # print(f'Skipped {example["id"]}')
                    continue

            logic_problem = sample['logic_problem']
            _, status, error = self.safe_execute_program(logic_problem)

            if status == 'parsing error' and error:

                try:
                    
                    correction = self.parsing_correction_generator(logic_problem, error)
                    refined_problem_string = json.dumps(logic_problem, ensure_ascii=False).replace(error, correction)
                    refined_problem = json.loads(refined_problem_string)

                except Exception as e:
                    error_message = f'Exception for {sample["id"]} for parsing response generation: {traceback.format_exc()}'
                    logger.error(error_message)
                    send_notification(error_message, "self_refinement.py sample error")
                    
                    refined_problem = logic_problem

                sample.update(
                    {
                        'refined_problem':refined_problem,
                        'skip': False
                    }
                )
                
                outputs.append(sample)
                
            # elif status == 'execution error' and error:
                
            #     full_prompt = self.execution_error_prompt[self.dataset_name](logic_problem, error)

            #     try:

            #         # open('this')

            #         response_string = self.refiner_api.generate(full_prompt, self.task_description_execution, {"type": "json_object"})

            #         response = json.loads(response_string)

            #         assert 'Correct Program' in response.keys(), 'Correct Program not in response.keys()'
            #         refined_problem = response['Correct Program']

            #         # programs = revised_program
            #     except Exception as e:
            #         print(f'Exception for {sample["id"]} for execution response: {e}')
            #         # print(traceback.format_exc())
            #         refined_problem = logic_problem

            #     output = {'id': sample['id'], 
            #             'context': sample['context'],
            #             'question': sample['question'], 
            #             'answer': sample['answer'],
            #             # 'options': example['options'],
            #             'raw_logic_programs': refined_problem,
            #               'skip': False}
            #     outputs.append(output)
            # elif status == 'success':
            #     sample.update({'skip':True})
            #     outputs.append(sample)

            # else:
                # print(f'Something went wrong with {example["id"]}. This sample will be skipped in the next iterations.')
                # sample.update({'skip':True})
                # outputs.append(sample)
                
            with open(save_file, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

        # save results


        # save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}_{self.prompt_mode}.json'

    
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
            verbose=False
        )
        
    try:    
    
        for round in range(starting_round, args.maximum_rounds+1):
            logger.info(f"Round {round} self-refinement")
            engine = SelfRefinementEngine(args, round, refiner = refiner)
            engine.single_round_self_refinement()
            
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        sys.exit(0)
            
    except Exception as e:
        
        error_message = f'A fatal error occurred: {traceback.format_exc()}'
        send_notification(error_message, "self_refinement.py fatal error")
        logger.error(error_message)
        sys.exit(0)
        
    logger.info("Finished Successfully")
    send_notification("Yippiee!", "self_refinement.py finished successfully")