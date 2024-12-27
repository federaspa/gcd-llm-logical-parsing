from utils.base import BasePrompter
from typing import Dict

class FOL_Prompter(BasePrompter):
    
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