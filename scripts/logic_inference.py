import os
import json
import argparse
import traceback
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from timeout_decorator import timeout, TimeoutError
import sys
import re

from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program

@dataclass
class InferenceConfig:
    model_name: Optional[str]
    dataset_name: str
    data_path: str
    split: str
    save_path: str
    programs_path: str
    self_refine_round: int = 0

class LogicProgram:
    """Wrapper class for handling logic program execution"""
    EXECUTOR_MAP = {
        'FOLIO': FOL_Prover9_Program,
        'FOLIOv2': FOL_Prover9_Program,
        'LogicNLI': FOL_Prover9_Program,
        'AR-LSAT': LSAT_Z3_Program
    }

    def __init__(self, dataset_name: str):
        if dataset_name not in self.EXECUTOR_MAP:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.executor_class = self.EXECUTOR_MAP[dataset_name]
        
    @staticmethod
    def parse_json(input_string: str) -> Dict:
        """Parse and clean JSON string"""
        # Remove markdown markers if present
        cleaned_string = re.sub(r'```json\n|```\n|```', '', input_string)
        # Remove whitespace and parse
        return json.loads(cleaned_string.strip())

    @timeout(seconds=60)
    def execute(self, logic_problem: Dict) -> Tuple[str, str, str]:
        
        try:
            parsed_problem = self.parse_json(logic_problem)
        except Exception as e:
            return 'N/A', 'json error', str(e)
        
        program = self.executor_class(parsed_problem)
        
        if not program.flag:
            return 'N/A', 'parsing error', program.formula_error
            
        answer, error_message = program.execute_program()
        
        if not program.flag:
            return 'N/A', 'parsing error', program.formula_error
            
        if answer is None:
            return 'N/A', 'execution error', error_message
            
        return program.answer_mapping(answer), 'success', ''

class LogicInferenceEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.program_executor = LogicProgram(config.dataset_name)

    def _get_file_name(self) -> str:
        """Generate the appropriate file name based on configuration"""
        prefix = f"self-refine-{self.config.self_refine_round}_" if self.config.self_refine_round > 0 else ""
        return f"{prefix}{self.config.dataset_name}_{self.config.split}_{self.config.model_name}.json"

    def load_logic_problems(self) -> List[dict]:
        """Load logic problems from file"""
        file_path = Path(self.config.programs_path) / self._get_file_name()
        with open(file_path) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.config.split} split.")
        return dataset

    # @timeout(seconds=60)
    # def safe_execute_program(self, logic_problem: Dict) -> Tuple[str, str, str]:
    #     """Safely execute a logic program with error handling"""
    #     try:
    #         parsed_problem = self.parse_json(logic_problem['raw'])
    #     except Exception as e:
    #         return 'N/A', 'json error', str(e)
        
    #     return self.program_executor.execute(parsed_problem)

    def process_samples(self, samples: List[Dict], problem_key: str) -> Tuple[int, int]:
        """Process a batch of samples for a specific problem type"""
        parsing_errors = 0
        execution_errors = 0

        for sample in tqdm(samples):
            if problem_key not in sample:
                continue

            try:
                logic_problem = sample[problem_key]['raw']
                answer_pred, status, error = self.program_executor.execute(logic_problem)
                # answer_pred, status, error = self.safe_execute_program(logic_problem)
                
                if status == 'parsing error':
                    parsing_errors += 1
                elif status == 'execution error':
                    execution_errors += 1
                    
                sample[problem_key].update({
                    'answer': sample['nl_problem']['answer'],
                    'predicted_answer': answer_pred,
                    'status': status,
                    'error': error
                })
                
            except TimeoutError:
                execution_errors += 1

        return parsing_errors, execution_errors

    def run(self):
        """Main execution method"""
        # Create save directory if it doesn't exist
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        outputs = self.load_logic_problems()
        
        # Process each type of logic problem
        for key in ['logic_problem', 'logic_problem_json', 'logic_problem_gcd']:
            print(f'\n{key.capitalize()}')
            parsing_errors, execution_errors = self.process_samples(outputs, key)
            print(f"Parsing errors: {parsing_errors}")
            print(f"Execution errors: {execution_errors}")
        
        # Save results
        save_file = save_path / self._get_file_name()
        with open(save_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        self.cleanup()

    @staticmethod
    def cleanup():
        """Clean up temporary files"""
        compiled_krb_dir = Path('./models/compiled_krb')
        if compiled_krb_dir.exists():
            print('Removing compiled_krb')
            os.system(f'rm -rf {compiled_krb_dir}')

def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save-path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--programs-path', type=str, default='./outputs/logic_problems')
    parser.add_argument('--self-refine-round', type=int, default=0)
    
    args = parser.parse_args()
    return InferenceConfig(**vars(args))

def get_available_models(config_path: str = './configs/models') -> List[str]:
    """Get list of available models from config directory"""
    config_dir = Path(config_path)
    return [f.stem for f in config_dir.glob('*.yml')]

def main():
    config = parse_args()
    
    # Get list of models to process
    models = [config.model_name] if config.model_name else get_available_models()
    
    print(f"Dataset: {config.dataset_name}")
    print(f"Models: {models}")
    print(f"Self-refine-round: {config.self_refine_round}")
    print(f"Split: {config.split}")
    print(f"Save path: {config.save_path}")
    
    for model in models:
        print(f"\nProcessing model: {model}")
        config.model_name = model
        engine = LogicInferenceEngine(config)
            
        try:
            engine.run()
        except KeyboardInterrupt:
            sys.exit(0)
        except FileNotFoundError as e:
            print(f'No such file or directory: {e}')

if __name__ == "__main__":
    main()