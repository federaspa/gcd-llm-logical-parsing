from dataclasses import dataclass
from typing import Dict, List, Callable, Any
import json


@dataclass
class Config:
    sketcher_name: str
    dataset_name: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    n_gpu_layers: int 
    n_ctx: int
    max_tokens: int
    n_threads: int
    stop_time: str|None
    timeout_seconds: int|None
    two_steps: bool = False
    force_unconstrained: bool = False
    force_constrained:bool = False
    debug: bool = False

class FOL_Prompter:
    def __init__(self, config: Config, templates: Dict[str,str]):
        self.config = config
        self.templates = templates

    def _get_predicates(self, constructs: Dict) -> str:
        
        predicates = constructs.get('fol_preds', None)
        
        if predicates:
            return ' | '.join(f'"{pred.split("(")[0]}"' for pred in predicates)
        else:
            return '[A-Z][a-z0-9]{2,15}'

        
    def _get_constants(self, constructs: Dict) -> str:
        
        constants = constructs.get('fol_consts', None)

        if constants:
            return' | '.join(f'"{con}"' for con in constants)            
        else:
            return'[a-z0-9]{2,15}'
        
    def get_grammar(self, constructs: dict) -> str:
        
        predicates = self._get_predicates(constructs)
        constants = self._get_constants(constructs)
                    
                            
        return self.templates['constrained_grammar'].replace('[[PREDICATES]]', predicates).replace('[[CONSTANTS]]', constants)
    
    def unconstrained(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.templates['unconstrained'].replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
    
    def constrained(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.templates['constrained'].replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)

    def json_wrap(self, unconstrained: str) -> str:
        return self.templates['json'].replace('[[unconstrained]]', unconstrained)
    
    def extract_constructs(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.templates['predicates'].replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
    
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
        
        
        
class SAT_Prompter:
    def __init__(self, config: Config, templates: Dict[str,str]):
        self.config = config
        self.templates = templates
        
    def unconstrained(self, sample: Dict) -> str:
        problem = sample['context']
        question = sample['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in sample['options']]).strip()
        
        full_prompt = self.templates['unconstrained'].replace('[[nl_problem]]', problem).replace('[[nl_question]]', question)
        full_prompt = full_prompt.replace('[[choices]]', choices_str)
        
        return full_prompt
    
    def constrained(self, sample: Dict) -> str:
        problem = sample['context']
        question = sample['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in sample['options']]).strip()
        
        full_prompt = self.templates['constrained'].replace('[[nl_problem]]', problem).replace('[[nl_question]]', question)
        full_prompt = full_prompt.replace('[[choices]]', choices_str)
        
        return full_prompt
    
    def json_wrap(self, unconstrained: str) -> str:
        
        full_prompt = self.templates['json'].replace('[[unconstrained]]', unconstrained).replace(r'\`\`\`.*', '')
        
        return full_prompt
    
    def parsing_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> tuple[str, str | None]:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        full_prompt, grammar = '', ''

        return full_prompt, grammar

    def get_grammar(self, logic_problem: dict) -> str:
        
        return ''

    def execution_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> str:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        problem_string = json.dumps(logic_problem)

        if mode == 'reasoning':
            return ''
        elif mode == 'generation':
            return ''
        
        
class LP_Prompter:
    def __init__(self, config: Config, templates: Dict[str,str]):
        self.config = config
        self.templates = templates
        
    def unconstrained(self, sample: Dict) -> str:
        problem = sample['context']
        question = sample['question'].strip()
        
        full_prompt = self.templates['unconstrained'].replace('[[nl_problem]]', problem).replace('[[nl_question]]', question)
        
        return full_prompt
    
    def constrained(self, sample: Dict) -> str:
        problem = sample['context']
        question = sample['question'].strip()
        
        full_prompt = self.templates['constrained'].replace('[[nl_problem]]', problem).replace('[[nl_question]]', question)
        
        return full_prompt
    
    def json_wrap(self, unconstrained: str) -> str:
        
        full_prompt = self.templates['json'].replace('[[unconstrained]]', unconstrained).replace(r'```\w+', '')
        
        return full_prompt
    
    def parsing_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> tuple[str, str | None]:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        full_prompt, grammar = '', ''

        return full_prompt, grammar

    def get_grammar(self, logic_problem: dict) -> str:
        
        return ''

    def execution_prompt(self, mode: str, logic_problem: dict, error: str, reasoning: str | None = None) -> str:
        assert mode in ['reasoning', 'generation'], 'wrong or no prompting mode specified'

        problem_string = json.dumps(logic_problem)

        if mode == 'reasoning':
            return ''
        elif mode == 'generation':
            return ''