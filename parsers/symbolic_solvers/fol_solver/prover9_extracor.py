import re
import json
from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula
from .Formula import FOL_Formula

# set the path to the prover9 executable
# os.environ['PROVER9'] = '../Prover9/bin'
os.environ['PROVER9'] = './models/symbolic_solvers/Prover9/bin'

class FOL_Prover9_Program:
    def __init__(self, logic_program:str) -> None:
        self.logic_program = logic_program
        self.flag, self.parsing_error_message = self.parse_logic_program()

    def parse_logic_program(self):
        try:        
               
            premises_string = self.logic_program.split("First-Order-Logic Question:")[0].split("First-Order-Logic Premises:")[1].strip()
            conclusion_string = self.logic_program.split("First-Order-Logic Question:")[1].strip()

            # Extract each premise and the conclusion using regex
            premises = premises_string.strip().split('\n')
            conclusion = conclusion_string.strip().split('\n')
                
            premises = [premise for premise in premises if re.sub(r'(?:[\',\",\`,\-,\s*])','', premise)]
            conclusion = [conc for conc in conclusion if re.sub(r'(?:[\',\"\`,\-,\s*])','', conc)]
            
            self.logic_premises = [premise.split(':::')[0].strip() for premise in premises]
            self.logic_conclusion = conclusion[0].split(':::')[0].strip()
            

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    self.prover9_premises.append('[[ERROR]]')
                    # print(f'error: {premise}')
                    continue
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                # print(f'error: {self.logic_conclusion}')
                self.prover9_conclusion = '[[ERROR]]'
            else:
                self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True, ''
        
        except Exception as e:
            print()
            print()
            print(self.logic_program)
            print()
            print(e)
            print()
            print()
            return False, str(e)

    def execute_program(self):
        try:
            
            assumptions = []
            
            goal = Expression.fromstring(self.prover9_conclusion)
            
            for premise in self.prover9_premises:
                
                if premise != '[[ERROR]]':
                    assumptions.append(Expression.fromstring(premise))
                else:
                    assumptions.append('[[ERROR]]')
            
            return str(goal), [str(a) for a in assumptions]
 
        except Exception as e:
            return None, None
        
if __name__ == "__main__":
    pass