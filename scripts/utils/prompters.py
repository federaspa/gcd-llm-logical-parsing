from utils.base import BasePrompter
from typing import Dict

class FOL_Prompter(BasePrompter):
    def _get_predicates(self, constructs: Dict) -> str:
        predicates = constructs.get('fol_preds', None)
        
        if predicates:
            return ' | '.join(f'"{pred.split("(")[0]}"' for pred in predicates)
        elif "symbolic" in self.config.dataset_name:
            return '[A-Z][0-9]'
        else:
            return '[A-Z][a-zA-Z0-9]+'

    def _get_constants(self, constructs: Dict) -> str:
        constants = constructs.get('fol_consts', None)

        if constants:
            return ' | '.join(f'"{con}"' for con in constants)            
        elif "symbolic" in self.config.dataset_name:
            return '[a-z][0-9]'
        else:
            return '[a-z0-9]+'
        
    def build_grammar(self, constructs: dict) -> str:
        predicates = self._get_predicates(constructs)
        constants = self._get_constants(constructs)
        return self.templates['constrained_grammar'].replace('[[PREDICATES]]', predicates).replace('[[CONSTANTS]]', constants)