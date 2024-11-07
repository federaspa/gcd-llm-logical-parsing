import sys
import subprocess
import tempfile
import os
from z3 import Solver

class LSAT_Z3_Program:
    def __init__(self, logic_program:dict) -> None:
        self.logic_program = logic_program
        self.formula_error = ''
        try:
            self.parse_logic_program()
            self.standard_code = self.to_standard_code()
        except Exception as e:
            print(e)
            self.standard_code = None
            self.flag = False
            return
        
        self.flag = True

        # create the folder to save the Pyke program
        cache_dir = os.path.join(os.path.dirname(__file__), '.cache_program')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def parse_logic_program(self):

        self.declarations = self.logic_program.get('declarations', [])
        self.constraints = self.logic_program.get('constraints', [])
        self.options = self.logic_program.get('options', [])

        return True

    def to_standard_code(self):

        standard_code = '(set-logic ALL)\n\n'

        standard_code += '; Declarations\n\n'
        for declaration in self.declarations:
            standard_code += f'{declaration}\n'
        
        standard_code += '\n; Constraints\n\n'
        for constraint in self.constraints:
            standard_code += f'(assert {constraint})\n\n'
            
        standard_code += '\n; Options\n\n'
        for option in self.options:
            standard_code += f'(push)\n(assert {option})\n(check-sat)\n(pop)\n\n'

        return standard_code
    
    def execute_program(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2') as f:
            f.write(self.standard_code)
            f.flush()
            
            # Create a solver and read the file
            solver = Solver()
            result = []
            
            process = subprocess.run(['z3', f.name], capture_output=True, text=True)
            
            # Parse the output
            output_lines = process.stdout.strip().split('\n')
            result = [line for line in output_lines if line in ['sat', 'unsat']]
            
        
        if len(result) == 0:
            return None, 'No Output'
        
        return result, ""
    
    def answer_mapping(self, answer):
        return answer

if __name__=="__main__":
    logic_program = {"declarations": ["people = EnumSort([Vladimir, Wendy])", "meals = EnumSort([breakfast, lunch, dinner, snack])", "foods = EnumSort([fish, hot_cakes, macaroni, omelet, poached_eggs])", "eats = Function([people, meals] -> [foods])"],
"constraints": ["ForAll([m:meals], eats(Vladimir, m) != eats(Wendy, m))", "ForAll([p:people, f:foods], Count([m:meals], eats(p, m) == f) <= 1)", "ForAll([p:people], Or(eats(p, breakfast) == hot_cakes, eats(p, breakfast) == poached_eggs, eats(p, breakfast) == omelet))", "ForAll([p:people], Or(eats(p, lunch) == fish, eats(p, lunch) == hot_cakes, eats(p, lunch) == macaroni, eats(p, lunch) == omelet))", "ForAll([p:people], Or(eats(p, dinner) == fish, eats(p, dinner) == hot_cakes, eats(p, dinner) == macaroni, eats(p, dinner) == omelet))", "ForAll([p:people], Or(eats(p, snack) == fish, eats(p, snack) == omelet))", "eats(Wendy, lunch) == omelet"],
"options": ["is_valid(Exists([m:meals], eats(Vladimir, m) == fish))", "is_valid(Exists([m:meals], eats(Vladimir, m) == hot_cakes))", "is_valid(Exists([m:meals], eats(Vladimir, m) == macaroni))", "is_valid(Exists([m:meals], eats(Vladimir, m) == omelet))", "is_valid(Exists([m:meals], eats(Vladimir, m) == poached_eggs))"]}

    z3_program = LSAT_Z3_Program(logic_program)
    print(z3_program.standard_code)

    output, error_message = z3_program.execute_program()
    print(output)
    print(z3_program.answer_mapping(output))