import os
import json
import argparse
import traceback
from typing import Tuple, List
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
import sys
import re

from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program

class LogicInferenceEngine:
    def __init__(self, args, model_name):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = model_name
        self.programs_path = args.programs_path
        self.save_path = args.save_path
        self.self_refine_round = args.self_refine_round
        
        program_executor_map = {
            'FOLIO': FOL_Prover9_Program, 
            'FOLIOv2': FOL_Prover9_Program, 
                                'LogicNLI': FOL_Prover9_Program,
                                'AR-LSAT': LSAT_Z3_Program
                                }
        
        self.program_executor = program_executor_map[self.dataset_name]
        
    def load_logic_problems(self) -> List[dict]:
        
        if self.self_refine_round > 0:
            programs_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.model_name}.json'
        else:
            programs_file = f'{self.dataset_name}_{self.split}_{self.model_name}.json'
        with open(os.path.join(self.programs_path, programs_file)) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset



    def load_json(self, input_string):
        """
        Preprocess a FOL (First Order Logic) JSON string to make it valid for parsing.
        
        Args:
            input_string (str): The input JSON string containing FOL expressions
            
        Returns:
            dict: The parsed JSON data as a Python dictionary
        """
        # Remove the ```json and ``` markers if present
        cleaned_string = re.sub(r'```json\n|```\n|```', '', input_string)
        
        # Remove any leading/trailing whitespace
        cleaned_string = cleaned_string.strip()
        
        # Parse the cleaned string as JSON
        json_data = json.loads(cleaned_string)
        return json_data


    @timeout(seconds=60)
    def safe_execute_program(self, logic_problem:str) -> Tuple[str,str, str]:
        
        try:
            logic_program = self.load_json(logic_problem['raw'])
        except Exception as e:
            return 'N/A', 'json error', str(e)
        
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
        
        outputs = self.load_logic_problems()

        
        if self.self_refine_round > 0:
            save_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.model_name}.json'
        else:
            save_file = f'{self.dataset_name}_{self.split}_{self.model_name}.json'
            
        for key in ['logic_problem', 'logic_problem_json', 'logic_problem_gcd']:
            
            print(f'\n{key.capitalize()}')
            
            parsing_error_count = 0
            execution_error_count = 0

            for sample in tqdm(outputs):
                
                answer = sample['nl_problem']['answer']
                

                if key in sample.keys():
                    
                    try:
                
                        logic_problem = sample.get(key, {})
                        answer_pred, status, error = self.safe_execute_program(logic_problem)
                        
                        if status == 'parsing error':
                            parsing_error_count += 1
                        elif status == 'execution error':
                            execution_error_count += 1
                            
                        sample[key].update({
                            'answer': answer,
                            'predicted_answer': answer_pred,
                            'status': status,
                            'error': error
                        })
                        
                    except TimeoutError:
                        execution_error_count += 1
                        pass
                
            print(f"Parsing: {parsing_error_count}")
            print(f"Execution: {execution_error_count}")
                
                
        # outputs.append(sample)
        
        with open(os.path.join(self.save_path, save_file), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    
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
    
    if args.model_name:
        models = [args.model_name]
    else:
        config_path = './configs/models'
        models = [os.path.splitext(f)[0] for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]

    print(f"Dataset: {args.dataset_name}")
    print(f"Models: {models}")
    print(f"Self-refine-round: {args.self_refine_round}")
    print(f"Split: {args.split}")
    print(f"Save path: {args.save_path}")
    
    
    for model in models:
        print(f"Models: {model}")
        engine = LogicInferenceEngine(args, model)
            
        try:
            engine.inference_on_dataset()
        
        except KeyboardInterrupt:
            sys.exit(0)
            
        except FileNotFoundError as e:
            print(f'No such file or directory: {e}')
        # send_notification("Yippiee!", "logic_inference.py finished successfully")