import json
import os
import sys
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from pathlib import Path
from pprint import pp
from timeout_decorator import timeout, TimeoutError

from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
<<<<<<< HEAD:models/refinement.py
from utils import OSModel, get_logger, send_notification, calculate_perplexity
from prompters import FOL_Prompter, LP_Prompter
=======
from scripts.utils.utils import OSModel, get_logger, send_notification, calculate_perplexity
from scripts.utils.prompters import FOL_Prompter, LP_Prompter
>>>>>>> 2bc58cbef31857f1b697f56f2da4554c232982ca:scripts/refinement.py

import traceback

@dataclass
class Config:
    sketcher_name: str
    refiner_name: str
    dataset_name: str
    models_path: str
    save_path: str
    split: str
    starting_round: int
    maximum_rounds: int
    n_gpu_layers: int
    verbose: bool
    gcd: bool
    static_preds: bool
    static_consts: bool
    
script_name = Path(__file__).stem
logger, log_file_name = get_logger(script_name)

class PromptGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.type = self._get_type()
        self._load_templates()
        self.prompter = self._get_prompter()

    def _get_type(self) -> str:
        types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'LogicNLI': 'FOL',
            'ProntoQA': 'LP',
            'ProofWriter': 'LP'
        }
        return types[self.config.dataset_name]
    
    def _get_prompter(self):
        prompters = {
            'FOL': FOL_Prompter,
            'LP': LP_Prompter
        }
        
        prompter = prompters[self.type](self.config, self.templates)
        return prompter

    def _load_templates(self):
        templates = {
            'parsing_user': f'./prompts/correction/{self.config.dataset_name}/parsing.txt',
            'execution_user': f'./prompts/correction/{self.config.dataset_name}/execution.txt',
            'parsing_reasoning_user': f'./prompts/correction/{self.config.dataset_name}/parsing_reasoning.txt',
            'execution_reasoning_user': f'./prompts/correction/{self.config.dataset_name}/execution_reasoning.txt',
            'grammar_file': f'./LLMs/grammars/{self.type}.gbnf',
            'json_grammar': './LLMs/grammars/json.gbnf',
            'prompt_template': 'prompts/prompt_templates/gemma.txt' if 'gemma' in self.config.sketcher_name else 'prompts/prompt_templates/llama.txt',
        }
        # Load and process templates
        self.templates = {}
        for key, path in templates.items():
            with open(path, 'r') as f:
                content = f.read()
                if 'user' in key:
                    with open(templates['prompt_template'], 'r') as pt:
                        prompt_template = pt.read()
                        content = prompt_template.replace('[[user]]', content)
                self.templates[key] = content


    def _read_file(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()
        


class SelfRefinementEngine(PromptGenerator):
    def __init__(self, config: Config, current_round: int, refiner: OSModel):
        super().__init__(config)
        self.current_round = current_round
        self.refiner_api = refiner
        self.sketcher_path = Path(config.models_path) / f"{config.sketcher_name}.gguf"
        self.refiner_path = Path(config.models_path) / f"{config.refiner_name}.gguf"
        self.gcd_dir = 'GCD' if config.gcd else 'NO_GCD'
        self.program_executor = FOL_Prover9_Program

    def _load_logic_problems(self) -> List[dict]:
        prefix = f'self-refine-{self.current_round-1}_' if self.current_round > 1 else ""
        
        if self.current_round == 1:
            programs_path = Path(self.config.save_path).parent / 'logic_problems' / f'{self.config.dataset_name}_{self.config.split}_{config.sketcher_name}.json'
        else:
            programs_path = Path(self.config.save_path) / self.gcd_dir / config.refiner_name / f'{prefix}{self.config.dataset_name}_{self.config.split}_{config.sketcher_name}.json'
        
        with open(programs_path) as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} examples from {self.config.split} split.")
        return dataset

    def _safe_execute_program(self, logic_program: dict) -> tuple[str, str, str]:
        program = self.program_executor(logic_program)
        if program.flag == False:
            return 'N/A', 'parsing error', program.formula_error
        answer, error_message = program.execute_program()
        if answer is None:
            return 'N/A', 'execution error', error_message
        return answer, 'success', ''
    
    @timeout(seconds=120)
    def _parsing_reasoning_generator(self, logic_problem: dict, error: str) -> str:
        user, _ = self.prompter.parsing(mode='reasoning', logic_problem=logic_problem, error=error)
        # logger.debug('REASONING GENERATION')
        response = self.refiner_api.invoke(
            prompt=user,
            temperature=1.0,
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity

    @timeout(seconds=120)
    def _parsing_correction_generator(self, logic_problem: dict, error: str, reasoning: str) -> str:
        user, grammar = self.prompter.parsing(mode='generation', logic_problem=logic_problem, error=error, reasoning=reasoning)
        # logger.debug('CORRECTION GENERATION')
        response = self.refiner_api.invoke(
            prompt=user,
            raw_grammar=grammar,
            temperature=0.5,
            top_p = 1,
            top_k=10,
            min_p=0.1,
            tfs_z=1,
            repeat_penalty=1,
        )
        
        content = response['choices'][0]['text']
        
        if not self.config.gcd:
            content = json.loads(response['choices'][0]['text']).get('corrt_formula', error)
        
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity
    
    @timeout(seconds=120)
    def _execution_reasoning_generation(self, logic_problem: dict, error: str) -> str:
        user = self.prompter.execution(mode='reasoning', logic_problem=logic_problem, error=error)
        response = self.refiner_api.invoke(
            prompt=user,
            temperature=1.0
        )
        
        content = response['choices'][0]['text']
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity

    @timeout(seconds=120)
    def _execution_correction_generation(self, logic_problem: dict, error: str, reasoning: str) -> dict:
        user = self.prompter.execution(mode='generation', logic_problem=logic_problem, error=error, reasoning=reasoning)
        response = self.refiner_api.invoke(
            prompt=user,
            raw_grammar=self.templates['json_grammar'],
            temperature=0.5,
        )
        
        content = json.loads(response['choices'][0]['text'])
        perplexity = calculate_perplexity(response['choices'][0]['logprobs'])
        
        return content, perplexity

    @timeout(seconds=6500)
    def single_round_self_refinement(self):
        logic_problems = self._load_logic_problems()
        
        save_path = Path(self.config.save_path) / self.gcd_dir / config.refiner_name
        save_file = save_path / f'self-refine-{self.current_round}_{self.config.dataset_name}_{self.config.split}_{config.sketcher_name}.json'
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        outputs = []

        if save_file.exists():
            with open(save_file, 'r') as f:
                outputs = json.load(f)

            existing_ids = {s["id"] for s in outputs}
            logic_problems = [s for s in logic_problems if s["id"] not in existing_ids]

        logger.info(f"{len(outputs)} already exist.\nLoaded {len(logic_problems)} examples from {self.config.split} split.")
        
        for sample in logic_problems:
            if sample.get('skip', False):
                outputs.append(sample)
                logger.info(f'Skipped {sample["id"]}')
                continue

            logic_problem:dict = sample.get('logic_problem', {})
            if not logic_problem:
                sample.update({'skip': True})
                outputs.append(sample)
                continue

            _, status, error = self._safe_execute_program(logic_problem)
            
            skip = False
            
            try:
                if status == 'parsing error' and error:
                    logger.info(f'Fixing parsing error for {sample["id"]}')
                    reasoning, reasoning_perplexity = self._parsing_reasoning_generator(logic_problem, error)
                    correction, correction_perplexity = self._parsing_correction_generator(logic_problem, error, reasoning)
                    
                    logic_problem.setdefault('parsing_errors', {})
                    logic_problem.setdefault('execution_errors', {})
                    
                    # logic_problem["parsing_errors"][error] = (correction, reasoning)
                    logic_problem["parsing_errors"][error] = {
                        'reasoning': reasoning,
                        'correction': correction,
                        'reasoning_perplexity': reasoning_perplexity,
                        'correction_perplexity': correction_perplexity
                    }
                    
                    
                    logic_problem["fol_rules"] = [correction if f == error else f for f in logic_problem["fol_rules"]]
                    
                elif status == 'execution error' and error:
                    logger.info(f'Fixing execution error for {sample["id"]}')
                    reasoning, reasoning_perplexity = self._execution_reasoning_generation(logic_problem, error)
                    logic_problem, correction_perplexity = self._execution_correction_generation(logic_problem, error, reasoning)
                    
                    logic_problem.setdefault('parsing_errors', {})
                    logic_problem.setdefault('execution_errors', {})
                    
                    logic_problem['execution_errors'][error] = {
                        'reasoning': reasoning,
                        'reasoning_perplexity': reasoning_perplexity,
                        'correction_perplexity': correction_perplexity
                    }


                else:
                    skip = True
            
            except Exception as e:
                error_message = f'Exception for {sample["id"]}: {traceback.format_exc()}'
                logger.error(error_message)
                skip = True

            
            sample.update({
                'logic_problem': logic_problem,
                'skip': skip
            })
            
            outputs.append(sample)

            with open(save_file, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)
                
        with open(save_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
                
        logger.info(f"Completed round {self.current_round} self-refinement")          

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--refiner-name', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--models-path', type=str, default='/data/users/fraspant/LLMs')
    parser.add_argument('--save-path', type=str, default='./outputs/refinement')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--starting-round', type=int, default=1)
    parser.add_argument('--maximum-rounds', type=int, default=3)
    parser.add_argument('--n-gpu-layers', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gcd', action='store_true')
    parser.add_argument('--static_preds', action='store_true')
    parser.add_argument('--static_consts', action='store_true')
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    config = parse_args()
    
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Sketcher: {config.sketcher_name}")
    logger.info(f"Refiner: {config.refiner_name}")
    logger.info(f"Grammar-Constrained: {config.gcd}")
    logger.info(f"Split: {config.split}")
    logger.info(f"Save path: {config.save_path}")
    
    refiner = OSModel(
        model_path= f"{config.models_path}/{config.refiner_name}.gguf",
        n_gpu_layers=config.n_gpu_layers,
        verbose=config.verbose
    )
    
    try:
        for round in range(config.starting_round, config.maximum_rounds + 1):
            logger.info(f"Round {round} self-refinement")
            engine = SelfRefinementEngine(config, round, refiner=refiner)
            engine.single_round_self_refinement()
            
    except KeyboardInterrupt:
        logger.error("KeyboardInterrupt")
        os.remove(log_file_name)
        sys.exit(0)
            
    except Exception as e:
        error_message = f'A fatal error occurred: {traceback.format_exc()}'
        send_notification(error_message, "self_refinement.py fatal error")
        logger.error(error_message)
        sys.exit(0)
        
    logger.info("Finished Successfully")
    send_notification("Yippiee!", "self_refinement.py finished successfully")