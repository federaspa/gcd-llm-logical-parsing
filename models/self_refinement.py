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

class PromptGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.sketcher_path = args.sketcher_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        # self.prompt_mode = args.prompt_mode
                        

        self.parsing_prompters: Callable = {'FOL': self.parsing_prompt_fol
                            # 'LogicNLI': self.parsing_prompt_folio
                            }
        
        
        self.execution_prompters: Callable = {'FOL': self.execution_prompt_fol,
                            # 'LogicNLI': self.execution_prompt_folio
                            }
        
        self.types = {
            'FOLIO': 'FOL',
            'FOLIOv2': 'FOL',
            'LogicNLI': 'FOL'
        }
        
        self.type = self.types[self.dataset_name]
        
        self.load_prompt_templates()
        
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

        self.parsing_template = self._read_file(templates['parsing_user'])
        self.execution_template = self._read_file(templates['execution_user'])
        self.parsing_reasoning_template = self._read_file(templates['parsing_reasoning_user'])
        self.execution_reasoning_template = self._read_file(templates['execution_reasoning_user'])
        
        self.parsing_system = self._read_file(templates['parsing_system'])
        self.execution_system = self._read_file(templates['execution_system'])
        
        self.grammar_template = self._read_file(templates['grammar_file'])

    def _read_file(self, path):
        with open(path, 'r') as f:
            return f.read()
    
    def parsing_prompt_fol(self, mode:str, logic_problem:dict, error:str, reasoning:str|None = None):
        
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        premises = '\n'.join(f for f in logic_problem['fol_rules'])
        conclusion = logic_problem['fol_conc']
        
        if mode == 'reasoning':
            full_prompt = self.parsing_reasoning_template.replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error)
            grammar = None
                
        elif mode == 'generation' and not self.gcd:
            
            full_prompt = self.parsing_template.replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning)
            grammar = None
            
            print('#'*50)
            print('#'*10, 'PROMPT', '#'*10)
            print('#'*50, '\n\n')

            print(full_prompt, '\n\n')

        elif mode == 'generation' and self.gcd:
            full_prompt = self.parsing_template.replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning)
            
            predicates = '\"' + logic_problem['fol_preds'][0].split('(')[0] + '\"'
            for pred in logic_problem['fol_preds'][1:]:
                predicates += ' | \"' + pred.split('(')[0] + '\"'
                
            grammar = self.grammar_template.replace('[[PREDICATES]]', predicates)

            print('#'*50)
            print('#'*10, 'PROMPT', '#'*10)
            print('#'*50, '\n\n')

            print(full_prompt, '\n\n')
        
        return full_prompt, grammar
    
    
    def execution_prompt_fol(self, mode:str, logic_problem:dict, error:str, reasoning:str|None = None):
        
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'
        
        problem_string = json.dumps(logic_problem)
        
        if mode == 'reasoning':
            return self.execution_reasoning_template.replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error)
        
        elif mode == 'generation':
            return self.execution_template.replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning)
    
   
    def load_dynamic_examples(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}_examples.json')) as f:
            dynamic_examples = json.load(f)
        return dynamic_examples

class SelfRefinementEngine(PromptGenerator):
    def __init__(self, args, current_round, refiner):
        
        # raise NotImplemented('missing execution fixing')
        
        #super init
        super().__init__(args)
        
        self.args = args
        
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.split = args.split
        # self.prompt_mode = args.prompt_mode
        
        self.current_round = current_round
        
        self.sketcher_name = os.path.splitext(args.sketcher_path)[0].split('/')[-1]
           
        self.refiner_api: OSModel = refiner
        self.refiner_name = os.path.splitext(args.refiner_path)[0].split('/')[-1]
        
        self.gcd = args.gcd
        self.gcd_dir = 'GCD' if self.gcd else 'NO_GCD'


        self.parsing_prompter:Callable = self.parsing_prompters[self.type]
        self.execution_prompter:Callable = self.execution_prompters[self.type]
        
        self.program_executor = FOL_Prover9_Program
                
        self.load_prompt_templates()
        
    def load_logic_problems(self) -> List[dict]:
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'            
            programs_path = os.path.join(self.save_path, self.gcd_dir, self.refiner_name, f'{prefix}{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        else:
            programs_path = os.path.join(self.save_path.replace('refinement', 'logic_problems'), f'{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
            
        with open(programs_path) as f:
            dataset = json.load(f)
        logger.info(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    
    def parsing_reasoning_generator(self, logic_problem:dict, error:str) -> str:
        
        user, _ = self.parsing_prompter(
            mode = 'reasoning', 
            logic_problem = logic_problem, 
            error = error)
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.parsing_system,
        )
        
        content = response['choices'][0]['message']['content']
        
        return content
    
    def parsing_correction_generator(self, logic_problem:dict, error:str, reasoning:str) -> str:
        
        user, grammar = self.parsing_prompter(
            mode = 'generation',
            logic_problem = logic_problem, 
            error = error,
            reasoning = reasoning)
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.parsing_system,
            raw_grammar=grammar
        )
        
        content = response['choices'][0]['message']['content']
        
        return content
    
    def execution_reasoning_generation(self, logic_problem:dict, error:str) -> str:
        
        user = self.execution_prompter(
            mode = 'reasoning',
            logic_problem = logic_problem,
            error = error
        )
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.parsing_system,
        )
        
        content = response['choices'][0]['message']['content']
        
        return content
    
    def execution_correction_generation(self, logic_problem:dict, error:str, reasoning:str) -> dict:
        
        user = self.execution_prompter(
            mode = 'generation',
            logic_problem = logic_problem,
            error = error,
            reasoning = reasoning
        )
        
        response = self.refiner_api.invoke(
            user=user,
            task_description=self.parsing_system,
            json_format=True
        )
        
        content = response['choices'][0]['message']['content']
        problem:dict = json.loads(content)
        
        problem['reasoning'] = reasoning
        
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
        
        save_path = os.path.join(self.save_path, self.gcd_dir, self.refiner_name)
        save_file = os.path.join(save_path, f'self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.sketcher_name}.json')
        
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
                    logger.info(f'Skipped {sample["id"]}')
                    continue

            logic_problem = sample['logic_problem']
            _, status, error = self.safe_execute_program(logic_problem)
            
            skip = False

            if status == 'parsing error' and error:
                
                logger.info(f'Fixing parsing error for {sample["id"]}')
                
                try:
                    
                    reasoning = self.parsing_reasoning_generator(logic_problem, error)
                    correction = self.parsing_correction_generator(logic_problem, error, reasoning)
                    # refined_problem_string = json.dumps(logic_problem, ensure_ascii=False).replace(error, correction)
                    # refined_problem = json.loads(refined_problem_string)
                    
                    print('#'*50)
                    print('#'*10, 'CORRECTION', '#'*10)
                    print('#'*50, '\n\n')

                    print(correction, '\n\n')
                    
                except Exception as e:
                    error_message = f'Exception for {sample["id"]} for parsing error correction: {traceback.format_exc()}'
                    logger.error(error_message)
                    send_notification(error_message, "self_refinement.py parsing correction error")
                    
                    refined_problem = logic_problem
                
            elif status == 'execution error' and error:
                
                logger.info(f'Fixing execution error for {sample["id"]}')
                
                
                try:
                    
                    reasoning = self.execution_reasoning_generation(logic_problem, error)
                    refined_problem = self.execution_correction_generation(logic_problem, error, reasoning)


                except Exception as e:
                    error_message = f'Exception for {sample["id"]} for execution error correction: {traceback.format_exc()}'
                    logger.error(error_message)
                    send_notification(error_message, "self_refinement.py execution correction error")
                    
                    refined_problem = logic_problem

            elif status == 'success':
                refined_problem = logic_problem
                skip = True
                
            else:
                refined_problem = None
                skip = True
                
            sample.update(
                {
                    'refined_problem':refined_problem,
                    'skip': skip
                }
            )
            
            outputs.append(sample)
                

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
    parser.add_argument('--save_path', type=str, default='./outputs/refinement')
    parser.add_argument('--split', type=str, default='dev')
    # parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    
    parser.add_argument('--starting_round', type=int, default=1)
    parser.add_argument('--maximum_rounds', type=int, default=3)
    
    parser.add_argument('--n_gpu_layers', type=int, default=0)
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    logger = get_logger(script_name)
    
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Sketcher: {args.sketcher_path}")
    logger.info(f"Refiner: {args.refiner_path}")
    # logger.info(f"Self-refine-round: {args.self_refine_round}")
    logger.info(f"Grammar-Constrained: {args.gcd}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Save path: {args.save_path}")
    
        
    refiner=OSModel(
            model_path=args.refiner_path,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False
        )
        
    try:    
    
        for round in range(args.starting_round, args.maximum_rounds+1):
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