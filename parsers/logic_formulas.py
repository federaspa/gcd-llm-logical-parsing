import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_extracor import FOL_Prover9_Program
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.csp_solver.csp_solver import CSP_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.programs_path = args.programs_path
        self.save_path = args.save_path
        self.prompt_mode = args.prompt_mode
        self.response_mode = args.response_mode

        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'FOLIOv2': FOL_Prover9_Program,
                                'ProntoQA': Pyke_Program, 
                                'ProofWriter': Pyke_Program}
        self.program_executor = program_executor_map[self.dataset_name]

    def load_logic_programs(self):
        
        programs_file = f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}.json'
        with open(os.path.join(self.programs_path, programs_file)) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
            
        save_file = f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}.json'
        
        with open(os.path.join(self.save_path, save_file), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program)
        # cannot parse the program
        if program.flag == False:
            return None, None, 'parsing error'
        # execuate the program
        goal, assumptions = program.execute_program()
        
        if not goal or not assumptions:
            return goal, assumptions, 'None goal or assumptions'
        
        return goal, assumptions, 'success'


    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # execute the logic program
            
            program = example['raw_logic_programs'][0].strip()
            
            goal, assumptions, flag = self.safe_execute_program(example['id'], program)
            if not flag == 'success':
                error_count += 1
                # print(error_message)

            # create output
            output = {'id': example['id'], 
                    'flag': flag,
                    'assumptions': assumptions,
                    'goal': goal}
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = './models/compiled_krb'
        if os.path.exists(complied_krb_dir):
            print('removing compiled_krb')
            os.system(f'rm -rf {complied_krb_dir}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str)
    parser.add_argument('--response_mode', type=str)
    parser.add_argument('--programs_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()