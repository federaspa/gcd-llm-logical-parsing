import re
import json
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula
from .Formula import FOL_Formula
from typing import Tuple

# set the path to the prover9 executable
# os.environ['PROVER9'] = '../Prover9/bin'
os.environ['PROVER9'] = 'scripts/symbolic_solvers/Prover9/bin'


class FOL_Prover9_Program:
    def __init__(self, logic_program:str, dataset_name='FOLIO') -> None:
        self.logic_program = logic_program
        self.flag, self.formula_error = self.parse_logic_program()
        self.dataset_name = dataset_name

    def parse_logic_program(self) -> Tuple[bool, str|None]:
        try:
            # Extract premises and conclusion
            premises = self.logic_program["fol_rules"]
            conclusion = self.logic_program["fol_conc"]
            
            if isinstance(premises, str):
                premises = premises.strip().split('\n')

            if isinstance(conclusion, list):
                assert len(conclusion) == 1, "There should be only one conclusion"
                conclusion = conclusion[0].strip()
                
            premises = [premise for premise in premises if re.sub(r'(?:[\'\"\`\-\s*])','', premise)]
            
            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = conclusion.split(':::')[0].strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise_index, premise in enumerate(self.logic_premises):
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:

                    return False, premise
                
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:

                return False, self.logic_conclusion
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True, None
        
        except Exception as e:
            return False, None
        
    def execute_program(self) -> Tuple[str|None, str]:
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            
            assumptions = []
            
            for a in self.prover9_premises:
                exp = Expression.fromstring(a)
                assumptions.append(exp)
                
            timeout = 10
            #prover = Prover9()
            #result = prover.prove(goal, assumptions)
            
            self.prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = self.prover.prove(verbose=False)
            # print(prover.proof())
            if result:
                return 'True', ''
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                # negation_result = prover.prove(negated_goal, assumptions)
                self.prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = self.prover.prove()
                if negation_result:
                    return 'False', ''
                else:
                    return 'Unknown', ''
        except Exception as e:
            # print(e)
            return None, str(e)
    
    @staticmethod
    def answer_mapping(answer:str) -> str:
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")
        
if __name__ == "__main__":
    
    logic_program = {
  "fol_preds": [
    "Dependent(x)",
    "Drinks(x)",
    "Jokes(x)",
    "Unaware(x)",
    "Student(x)"
  ],
  "fol_consts": [
    "rina"
  ],
  "fol_rules": [
    "∀x (Drinks(x) → Dependent(x))",
    "∀x (Drinks(x) ⊕ Jokes(x))",
    "∀x (Jokes(x) → ¬Unaware(x))",
    "(Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina))",
    "¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina))"
  ],
  "fol_conc": "Jokes(rina) ⊕ Unaware(rina)"
    }
    
    prover9_program = FOL_Prover9_Program(logic_program)
    
    # print('\n'.join(prover9_program.prover9_premises))
    # print(prover9_program.prover9_conclusion)
    answer, error_message = prover9_program.execute_program()
    # print(answer)
    # print(error_message)