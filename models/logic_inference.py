import os
import json
import argparse
import traceback
from typing import Tuple, List
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
import sys

from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.sketcher_name = args.sketcher_name
        self.programs_path = args.programs_path
        self.save_path = args.save_path
        self.self_refine_round = args.self_refine_round
        
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'LogicNLI': FOL_Prover9_Program,
                                'AR-LSAT': LSAT_Z3_Program
                                }
        
        self.program_executor = program_executor_map[self.dataset_name]
        
        self.dataset = self.load_logic_problems()

    def load_logic_problems(self) -> List[dict]:
        
        if self.self_refine_round > 0:
            programs_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}.json'
        else:
            programs_file = f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json'
        with open(os.path.join(self.programs_path, programs_file)) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

    @timeout(seconds=60)
    def safe_execute_program(self, logic_program:dict) -> Tuple[str,str, str]:
        program = self.program_executor(logic_program)
        # cannot parse the program
        if program.flag == False:
            return 'N/A', 'parsing error', program.formula_error
        # execuate the program
        answer, error_message = program.execute_program()
        
        if program.flag == False:
            return 'N/A', 'parsing error', program.formula_error
        
        # not executable
        if answer is None:
            return 'N/A', 'execution error', error_message
        # successfully executed
        return program.answer_mapping(answer), 'success', ''

    def inference_on_dataset(self):
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        outputs = []
        parsing_error_count = 0
        execution_error_count = 0
        
        parsing_error_count_constrained = 0
        execution_error_count_constrained = 0
        
        if self.self_refine_round > 0:
            save_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}.json'
        else:
            save_file = f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json'
            
            
        # if os.path.exists(os.path.join(self.save_path, save_file)):
        #     print(f"File {save_file} already exists. Skipping...")
        #     return
        
        for sample in tqdm(self.dataset):
            # execute the logic program
                           
            if 'logic_problem' in sample.keys():
                
                try:
                
                    logic_problem = sample.get('logic_problem', {})
                    
                    answer, status, error = self.safe_execute_program(logic_problem)
                    
                    if status == 'parsing error':
                        parsing_error_count += 1
                    elif status == 'execution error':
                        execution_error_count += 1
                        
                    sample['logic_problem'].update({
                        'answer': sample['answer'],
                        'predicted_answer': answer,
                        'status': status,
                        'error': error
                    })
                    
                except TimeoutError:
                    execution_error_count += 1
                    pass
                
            if 'logic_problem_gcd' in sample.keys():
                
                try:
                                
                    logic_problem_constrained = sample.get('logic_problem_gcd', {})
                    answer_constrained, status_constrained, error_constrained = self.safe_execute_program(logic_problem_constrained)

                    if status_constrained == 'parsing error':
                        parsing_error_count_constrained += 1
                        
                    elif status == 'execution error':
                        execution_error_count_constrained += 1
                        
                    sample['logic_problem_gcd'].update({
                        'answer': sample['answer'],
                        'predicted_answer': answer_constrained,
                        'status': status_constrained,
                        'error': error_constrained
                    })
                        
                except TimeoutError:
                    execution_error_count_constrained += 1
                    pass
                
            outputs.append(sample)
            
            with open(os.path.join(self.save_path, save_file), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        print("\nUNCONSTRAINED")
        print(f"Parsing: {parsing_error_count}")
        print(f"Execution: {execution_error_count}")
        print("\nCONSTRAINED")
        print(f"Parsing: {parsing_error_count_constrained}")
        print(f"Execution: {execution_error_count_constrained}")
            
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--programs-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--self-refine-round', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    print(f"Dataset: {args.dataset_name}")
    print(f"Sketcher: {args.sketcher_name}")
    print(f"Self-refine-round: {args.self_refine_round}")
    print(f"Split: {args.split}")
    print(f"Save path: {args.save_path}")
    
    engine = LogicInferenceEngine(args)
    
    try:
        engine.inference_on_dataset()
    
    except KeyboardInterrupt:
        sys.exit(0)
                
    except Exception as e:

        error_message = f"A fatal error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        sys.exit(0)
                
    print("Finished Successfully")
    # send_notification("Yippiee!", "logic_inference.py finished successfully")