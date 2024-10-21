import json
import os
import sys
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
from utils import OSModel, get_logger, send_notification

import traceback

@dataclass
class Config:
    sketcher_path: str
    refiner_path: str
    dataset_name: str
    data_path: str
    save_path: str
    split: str
    starting_round: int
    maximum_rounds: int
    n_gpu_layers: int
    verbose: bool
    gcd: bool

class PromptGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.type = self.get_type()
        self.load_prompt_templates()

    def get_type(self) -> str:
        types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'LogicNLI': 'FOL'
        }
        return types[self.config.dataset_name]

    def load_prompt_templates(self):
        templates = {
            'parsing_user': f'./prompts/correction/user/{self.type}_parsing.txt',
            'execution_user': f'./prompts/correction/user/{self.type}_execution.txt',
            'parsing_reasoning_user': f'./prompts/correction/user/{self.type}_parsing_reasoning.txt',
            'execution_reasoning_user': f'./prompts/correction/user/{self.type}_execution_reasoning.txt',
            'parsing_system': f'./prompts/correction/system/{self.type}_parsing.txt',
            'execution_system': f'./prompts/correction/system/{self.type}_execution.txt',
            'grammar_file': f'./LLMs/grammars/{self.type}.gbnf'
        }

        self.templates = {key: self._read_file(path) for key, path in templates.items()}

    def _read_file(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    def parsing_prompt_fol(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> tuple[str, str | None]:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        premises = '\n'.join(logic_problem['fol_rules'])
        conclusion = logic_problem['fol_conc']

        if mode == 'reasoning':
            full_prompt = self.templates['parsing_reasoning_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error)
            grammar = None
        elif mode == 'generation':
            full_prompt = self.templates['parsing_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')
            grammar = self.get_grammar(logic_problem) if self.config.gcd else None

        return full_prompt, grammar

    def get_grammar(self, logic_problem: dict) -> str:
        predicates = ' | '.join(f'"{pred.split("(")[0]}"' for pred in logic_problem['fol_preds'])
        return self.templates['grammar_file'].replace('[[PREDICATES]]', predicates)

    def execution_prompt_fol(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> str:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        problem_string = json.dumps(logic_problem)

        if mode == 'reasoning':
            return self.templates['execution_reasoning_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error)
        elif mode == 'generation':
            return self.templates['execution_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')

class SelfRefinementEngine(PromptGenerator):
    def __init__(self, config: Config, current_round: int, refiner: OSModel):
        super().__init__(config)
        self.current_round = current_round
        self.refiner_api = refiner
        self.sketcher_name = Path(config.sketcher_path).stem
        self.refiner_name = Path(config.refiner_path).stem
        self.gcd_dir = 'GCD' if config.gcd else 'NO_GCD'
        self.program_executor = FOL_Prover9_Program

    def load_logic_problems(self) -> List[dict]:
        prefix = f'self-refine-{self.current_round-1}_' if self.current_round > 1 else ""
        
        if self.current_round == 1:
            programs_path = Path(self.config.save_path).parent / 'logic_problems' / f'{self.config.dataset_name}_{self.config.split}_{self.sketcher_name}.json'
        else:
            programs_path = Path(self.config.save_path) / self.gcd_dir / self.refiner_name / f'{prefix}{self.config.dataset_name}_{self.config.split}_{self.sketcher_name}.json'
        
        with open(programs_path) as f:
            dataset = json.load(f)
        return dataset

    def safe_execute_program(self, logic_program: dict) -> tuple[str, str, str]:
        program = self.program_executor(logic_program)
        if program.flag == False:
            return 'N/A', 'parsing error', program.formula_error
        answer, error_message = program.execute_program()
        if answer is None:
            return 'N/A', 'execution error', error_message
        return answer, 'success', ''
    
    def parsing_reasoning_generator(self, logic_problem: dict, error: str) -> str:
        user, _ = self.parsing_prompt_fol(mode='reasoning', logic_problem=logic_problem, error=error)
        # logger.debug('REASONING GENERATION')
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.templates['parsing_system'],
            temperature=1.0,
        )
        return response['choices'][0]['message']['content']

    def parsing_correction_generator(self, logic_problem: dict, error: str, reasoning: str) -> str:
        user, grammar = self.parsing_prompt_fol(mode='generation', logic_problem=logic_problem, error=error, reasoning=reasoning)
        # logger.debug('CORRECTION GENERATION')
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.templates['parsing_system'],
            raw_grammar=grammar,
            temperature=0.5,
            top_p = 1,
            top_k=10,
            min_p=0.1,
            tfs_z=1,
            repeat_penalty=1,
        )
    
        # logger.debug(grammar)
        # logger.debug(response)
        
        return response['choices'][0]['message']['content']

    def execution_reasoning_generation(self, logic_problem: dict, error: str) -> str:
        user = self.execution_prompt_fol(mode='reasoning', logic_problem=logic_problem, error=error)
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.templates['parsing_system'],
            temperature=1.0
        )
        return response['choices'][0]['message']['content']

    def execution_correction_generation(self, logic_problem: dict, error: str, reasoning: str) -> dict:
        user = self.execution_prompt_fol(mode='generation', logic_problem=logic_problem, error=error, reasoning=reasoning)
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.templates['parsing_system'],
            json_format=True,
            temperature=0.5,
        )
        return json.loads(response['choices'][0]['message']['content'])

    def single_round_self_refinement(self):
        logic_problems = self.load_logic_problems()
        
        for sample in logic_problems:

            logic_problem:dict = sample.get('logic_problem', {})
            if not logic_problem:
                continue

            _, status, error = self.safe_execute_program(logic_problem)
            
            if status == 'parsing error' and error:
                
                nl = '\n'.join(r for r in sample['nl_problem']['nl_rules'])
                fol = '\n'.join(r for r in logic_problem['fol_rules'])
                
                print(f"Original NL:\n\n{nl}\n")
                print(f"Original FOL:\n\n{fol}\n")
                print("Error:", error)
                
                input("Press Enter to continue...")
                
                print("Reasoning...")
                reasoning = self.parsing_reasoning_generator(logic_problem, error)
                
                print(reasoning)
                
                print("Fixing...")
                correction = self.parsing_correction_generator(logic_problem, error, reasoning)
                

                print("Correction:", correction)    
                
            


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-path', type=str, required=True)
    parser.add_argument('--refiner-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--save-path', type=str, default='./outputs/refinement')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--starting-round', type=int, default=1)
    parser.add_argument('--maximum-rounds', type=int, default=3)
    parser.add_argument('--n-gpu-layers', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    config = parse_args()
    
    refiner = OSModel(
        model_path=config.refiner_path,
        n_gpu_layers=config.n_gpu_layers,
        verbose=config.verbose
    )
    
    try:
        for round in range(config.starting_round, config.maximum_rounds + 1):
            engine = SelfRefinementEngine(config, round, refiner=refiner)
            engine.single_round_self_refinement()
            
    except KeyboardInterrupt:
        sys.exit(0)