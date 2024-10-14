import os
import json
import argparse
import traceback
from typing import Tuple, List
from tqdm import tqdm
import sys

class ErrorAnnotator:
    def __init__(self, args):
        self.args = args
        
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.save_path:str = args.save_path
        self.split = args.split
        self.self_refine_round = args.self_refine_round
        
        self.sketcher_name = os.path.splitext(args.sketcher_path)[0].split('/')[-1]
           
        self.refiner_name = os.path.splitext(args.refiner_path)[0].split('/')[-1]
        
        self.gcd = args.gcd
        self.gcd_dir = 'GCD' if self.gcd else 'NO_GCD'

        self.dataset = self.load_logic_problems()

    def load_logic_problems(self) -> List[dict]:
        prefix = ""
        if self.self_refine_round > 0:
            prefix = f'self-refine-{self.self_refine_round}_'            
            programs_path = os.path.join(self.save_path.replace('annotated', 'refinement'), self.gcd_dir, self.refiner_name, f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        else:
            programs_path = os.path.join(self.save_path.replace('annotated', 'logic_problems'), f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        with open(programs_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

    def run(self):
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        prefix = "" if self.self_refine_round == 0 else f'self-refine-{self.self_refine_round}_'

        analysis_file = f"./analysis_results/{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}_error_types.json"

        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                error_types_global = json.load(f)
        else:
            error_types_global = {}
                
        save_file = f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}.json'
            
        error_types_round = {}
        
        outputs = []
            
        for sample in tqdm(self.dataset):
            # execute the logic program
            
            logic_problem = sample['logic_problem']
            
            if not logic_problem:
                continue
                        
            if not 'parsing_errors' in logic_problem.keys():
                continue
            
            parsing_errors = logic_problem["parsing_errors"]
            
            error_types = {} if not 'error_types' in sample.keys() else sample['error_types']
            
            for error, correction in parsing_errors.items():
                print(f"Error: {error}")
                print(f"Correction: {correction}")
                
                x = input("Error type: ")
                
                if not x in error_types:
                    error_types[x] = 1
                else:
                    error_types[x] += 1
                    
                if not x in error_types_round:
                    error_types_round[x] = 1
                else:
                    error_types_round[x] += 1
            
            sample['error_types'] = error_types
            error_types_global[f"round_{self.self_refine_round}"] = error_types_round
            
            outputs.append(sample)
            
            with open(os.path.join(self.save_path, save_file), 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
                
            with open(f"./analysis_results/{self.dataset_name}_{self.split}_{self.sketcher_name}_error_types.json", 'w') as f:
                json.dump(error_types_global, f, indent=2, ensure_ascii=False)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-path', type=str, required=True)
    parser.add_argument('--refiner-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--self-refine-round', type=int, required=True)
    
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--save-path', type=str, default='./outputs/annotated')
    parser.add_argument('--split', type=str, default='dev')

    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    engine = ErrorAnnotator(args)
    
    engine.run()