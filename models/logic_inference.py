import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.csp_solver.csp_solver import CSP_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
from backup_answer_generation import Backup_Answer_Generator

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.programs_path = args.programs_path
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy
        self.backup_path = args.backup_LLM_result_path
        self.prompt_mode = args.prompt_mode
        self.response_mode = args.response_mode
        self.self_refine_round = args.self_refine_round

        self.dataset = self.load_logic_programs()
        self.ground_truth = self.load_ground_truth()
        
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'FOLIOv2': FOL_Prover9_Program,
                                'ProntoQA': Pyke_Program, 
                                'ProofWriter': Pyke_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        
        self.backup_result_path = os.path.join(self.backup_path, f'{self.backup_strategy}_{self.dataset_name}_{self.split}_{self.model_name}.json')
        
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.backup_result_path)

    def load_ground_truth(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json'), 'r') as f:
            ground_truth_raw = json.load(f)
            
        ground_truth = {
            point['id']: {
                'context_fol': point['context_fol']
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

    def load_logic_programs(self):
        
        if self.self_refine_round > 0:
            programs_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}.json'
        else:
            programs_file = f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}.json'
        with open(os.path.join(self.programs_path, programs_file)) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.self_refine_round > 0:
            save_file = f'self-refine-{self.self_refine_round}_{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}_backup-{self.backup_strategy}.json'
        else:
            save_file = f'{self.dataset_name}_{self.split}_{self.model_name}_{self.prompt_mode}_{self.response_mode}_backup-{self.backup_strategy}.json'
        
        with open(os.path.join(self.save_path, save_file), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program, self.dataset_name, self.prompt_mode, self.response_mode)
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', program.formula_error, program.parsing_error_index
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'execution error', None, None
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', None, None

    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # execute the logic program
            
            program = example['raw_logic_programs'][0].strip()
            answer, flag, formula_error, error_index = self.safe_execute_program(example['id'], program)
            if not flag == 'success':
                error_count += 1
                # print(example['id'])

            # create output
            output = {'id': example['id'], 
                    # 'context': example['context'],
                    'question': example['question'], 
                    'answer': example['answer'],
                    'flag': flag,
                    # 'error': error_message,
                    'predicted_answer': answer}
            if formula_error:
                output.update({
                    'wrong_formula': formula_error,
                    'correct_formula': self.ground_truth[example['id']]['context_fol'][error_index]
                })
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
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='static')
    parser.add_argument('--response_mode', type=str, choices=['text', 'json'], default='text')
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument('--programs_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'Direct', 'CoT'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='./baselines/results')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()