from dataclasses import dataclass
from typing import Dict, List, Callable, Any
import json

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
    static_preds: bool
    static_consts: bool

class FOL_Prompter:
    def __init__(self, config: Config, templates: Dict[str,str]):
        self.config = config
        self.templates = templates

    def unstructured(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.templates['unstructured_user'].replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)

    def structured(self, unstructured: str) -> str:
        return self.templates['structured_user'].replace('[[unstructured]]', unstructured)

    def _get_predicates(self, logic_problem: str) -> str:
        if self.config.static_preds:
            return '[A-Z][a-z0-9]{2,15}'
        else:
            return ' | '.join(f'"{pred.split("(")[0]}"' for pred in logic_problem['fol_preds'])
        
    def _get_constants(self, logic_problem: str) -> str:
        if len(logic_problem['fol_consts']) > 0 and not self.config.static_consts:
            return' | '.join(f'"{con}"' for con in logic_problem['fol_consts'])
        else:
            return'[A-Z][a-z0-9]{2,15}'
        
    def get_grammar(self, logic_problem: dict) -> str:
        
        predicates = self._get_predicates(logic_problem)
        constants = self._get_constants(logic_problem)
                    
        return self.templates['grammar_file'].replace('[[PREDICATES]]', predicates).replace('[[CONSTANTS]]', constants)
    
    
    def parsing(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> tuple[str, str | None]:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        premises = '\n'.join(logic_problem['fol_rules'])
        conclusion = logic_problem['fol_conc']

        if mode == 'reasoning':
            full_prompt = self.templates['parsing_reasoning_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error)
            grammar = None
        
        elif mode == 'generation' and self.config.gcd:
            full_prompt = self.templates['parsing_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')
            grammar = self.get_grammar(logic_problem)
            
        elif mode == 'generation' and not self.config.gcd:
            full_prompt = self.templates['parsing_user'].replace('[[PREMISES]]', premises)
            full_prompt = full_prompt.replace('[[CONCLUSION]]', conclusion)
            full_prompt = full_prompt.replace('[[ERROR]]', error)
            full_prompt = full_prompt.replace('[[REASONING]]', reasoning or '')
            grammar = self.templates['json_grammar']
            
        return full_prompt, grammar


    def execution(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> str:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        problem_string = json.dumps(logic_problem)

        if mode == 'reasoning':
            return self.templates['execution_reasoning_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error)
        elif mode == 'generation':
            return self.templates['execution_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')
        
        
        
class LP_Prompter:
    def __init__(self, config: Config, templates: Dict[str,str]):
        self.config = config
        self.templates = templates
        
    def _get_predicates(self, logic_problem: str) -> str:
        if self.config.static_preds:
            return '[A-Z][a-z0-9]{2,15}'
        else:
            return ' | '.join(f'"{pred.split("(")[0]}"' for pred in logic_problem['fol_preds'])
        
    def _get_constants(self, logic_problem: str) -> str:
        if self.config.static_consts:
            return'[A-Z][a-z0-9]{2,15}'
        else:
            return' | '.join(f'"{con}"' for con in logic_problem['fol_consts'])

    def parsing_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> tuple[str, str | None]:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        premises = '\n'.join(logic_problem['fol_rules'])
        conclusion = logic_problem['fol_conc']

        if mode == 'reasoning':
            full_prompt = self.templates['parsing_reasoning_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error)
            grammar = None
        elif mode == 'generation':
            full_prompt = self.templates['parsing_user'].replace('[[PREMISES]]', premises).replace('[[CONCLUSION]]', conclusion).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')
            grammar = self.get_grammar(logic_problem) if self.config.gcd else None

            # raise Exception(grammar)

        return full_prompt, grammar

    def get_grammar(self, logic_problem: dict) -> str:
        
        predicates = self._get_predicates(logic_problem)
        constants = self._get_constants(logic_problem)
                    
        return self.templates['grammar_file'].replace('[[PREDICATES]]', predicates).replace('[[CONSTANTS]]', constants)

    def execution_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> str:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        problem_string = json.dumps(logic_problem)

        if mode == 'reasoning':
            return self.templates['execution_reasoning_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error)
        elif mode == 'generation':
            return self.templates['execution_user'].replace('[[PROBLEM]]', problem_string).replace('[[ERROR]]', error).replace('[[REASONING]]', reasoning or '')