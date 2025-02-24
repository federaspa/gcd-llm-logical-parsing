from typing import Dict
from abc import ABC, abstractmethod
from utils.template_manager import TemplateManager
from typing import Dict

# class BasePrompter(ABC):


#     @abstractmethod
#     def build_grammar(self) -> str:
#         pass

#     def format_prompt(self, sample: Dict) -> str:
#         problem = '\n'.join(sample['context'])
#         question = sample['question'].strip()
#         return self.template_manager.prompt_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)

class FOL_Prompter():
    
    def __init__(self, config: dict):
        self.config = config
        self.template_manager = TemplateManager(config)
    
    def _get_predicates(self) -> str:
        if "symbolic" in self.config.dataset_name:
            return '[A-Z][0-9]'
        else:
            return '[A-Z][a-zA-Z0-9]+'

    def _get_constants(self) -> str:
        if "symbolic" in self.config.dataset_name:
            return '[a-z][0-9]'
        else:
            return '[a-z0-9]+'
        
    def build_grammar(self, template_key) -> str:
        
        predicates = self._get_predicates()
        constants = self._get_constants()
                
        if not self.template_manager.grammar_templates[template_key]:
            return None
                
        return self.template_manager.grammar_templates[template_key].replace('[[PREDICATES]]', predicates).replace('[[CONSTANTS]]', constants)
    
    def format_prompt(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.template_manager.prompt_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)
    
class SIN_Prompter():
    def __init__(self, config: dict):
        self.config = config
        self.template_manager = TemplateManager(config)
    
    def build_grammar(self, template_key) -> str:
        
        if not self.template_manager.grammar_templates[template_key]:
            return None
                
        return self.template_manager.grammar_templates[template_key]
    
    def format_prompt(self, sample: Dict) -> str:
        question = sample['question'].strip()
        return self.template_manager.prompt_template.replace('[[nl_problem]]', question)
    
    
class LD_Prompter():
    def __init__(self, config: dict):
        self.config = config
        self.template_manager = TemplateManager(config)
    
    def build_grammar(self, template_key) -> str:
        
        if not self.template_manager.grammar_templates[template_key]:
            return None
                
        return self.template_manager.grammar_templates[template_key]
    
    def format_prompt(self, sample: Dict) -> str:
        problem = '\n'.join(sample['context'])
        question = sample['question'].strip()
        return self.template_manager.prompt_template.replace('[[nl_problem]]', problem).replace('[[nl_conclusion]]', question)