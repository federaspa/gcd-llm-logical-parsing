import ast
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re

@dataclass
class EnumType:
    name: str
    values: List[str]

@dataclass
class FunctionDecl:
    name: str
    args: List[Tuple[str, str]]  # [(arg_name, type_name), ...]
    return_type: str

class Z3toSMTLIB:
    def __init__(self):
        self.enum_types: Dict[str, EnumType] = {}
        self.functions: Dict[str, FunctionDecl] = {}
        self.constraints: List[str] = []
        self.option_checks: List[str] = []
        
    def parse_enum_sort(self, line: str) -> None:
        """Parse EnumSort declaration and convert to SMT-LIB2 format"""
        # Match pattern: name_sort, (val1, val2) = EnumSort('name', ['val1', 'val2'])
        match = re.match(r"(\w+)_sort,\s*\(([\w,\s]+)\)\s*=\s*EnumSort\('(\w+)',\s*\[(.*?)\]\)", line)
        if match:
            sort_var, values_tuple, sort_name, values_list = match.groups()
            values = [v.strip().strip("'") for v in values_list.split(',')]
            self.enum_types[sort_var] = EnumType(sort_name, values)
            
    def parse_function(self, line: str) -> None:
        """Parse Function declaration and convert to SMT-LIB2 format"""
        # Match pattern: func = Function('func', type1, type2, return_type)
        match = re.match(r"(\w+)\s*=\s*Function\('(\w+)',\s*(.*?)\)", line)
        if match:
            func_var, func_name, type_list = match.groups()
            types = [t.strip() for t in type_list.split(',')]
            return_type = types[-1]
            arg_types = types[:-1]
            self.functions[func_var] = FunctionDecl(
                func_name,
                [(f"arg{i}", t.replace('_sort', '')) for i, t in enumerate(arg_types)],
                return_type.replace('_sort', '')
            )

    def generate_smtlib2(self, python_code: str) -> str:
        """Convert Z3 Python code to SMT-LIB2 format"""
        lines = python_code.split('\n')
        smtlib_parts = []
        
        # Parse declarations
        smtlib_parts.append("; Declarations")
        for line in lines:
            line = line.strip()
            if 'EnumSort' in line:
                self.parse_enum_sort(line)
            elif 'Function' in line:
                self.parse_function(line)
                
        # Generate enum type declarations
        for enum_type in self.enum_types.values():
            decl = f"(declare-datatypes () (({enum_type.name} {' '.join(enum_type.values)})))"
            smtlib_parts.append(decl)
            
        # Generate function declarations
        smtlib_parts.append("\n; Function declarations")
        for func in self.functions.values():
            arg_types = ' '.join(f"({arg[0]} {arg[1]})" for arg in func.args)
            decl = f"(declare-fun {func.name} ({arg_types}) {func.return_type})"
            smtlib_parts.append(decl)
            
        # Generate helper functions for counting
        smtlib_parts.append("""
; Helper function for counting occurrences
(define-fun count-food ((p People) (f Foods)) Int
  (+ (ite (= (eats p breakfast) f) 1 0)
     (ite (= (eats p lunch) f) 1 0)
     (ite (= (eats p dinner) f) 1 0)
     (ite (= (eats p snack) f) 1 0)))""")
        
        # Convert constraints
        in_constraints = False
        smtlib_parts.append("\n; Constraints")
        for line in lines:
            line = line.strip()
            if line.startswith('pre_conditions.append'):
                constraint = self._convert_constraint(line)
                if constraint:
                    smtlib_parts.append(f"(assert {constraint})")
                    
        # Convert option checks
        smtlib_parts.append("\n; Option checks")
        checking_options = False
        current_option = 'A'
        for line in lines:
            line = line.strip()
            if line.startswith('if is_valid('):
                check = self._convert_option_check(line, current_option)
                if check:
                    smtlib_parts.append(check)
                    current_option = chr(ord(current_option) + 1)
                    
        # Add final commands
        smtlib_parts.append("\n; Get model if needed")
        smtlib_parts.append("(get-model)")
        
        return '\n'.join(smtlib_parts)
    
    def _convert_constraint(self, line: str) -> str:
        """Convert a single Z3 constraint to SMT-LIB2 format"""
        # Extract the constraint part from pre_conditions.append(...)
        constraint = line[line.index('(')+1:line.rindex(')')]
        
        # Convert ForAll
        if constraint.startswith('ForAll'):
            match = re.match(r"ForAll\(\[(.*?)\],\s*(.*)\)", constraint)
            if match:
                vars_part, body = match.groups()
                vars_decl = ' '.join(f"({v.split(':')[0].strip()} {v.split(':')[1].strip().replace('_sort', '')})"
                                   for v in vars_part.split(','))
                body = self._convert_expression(body)
                return f"(forall ({vars_decl}) {body})"
        
        # Convert direct equality
        elif '==' in constraint:
            left, right = constraint.split('==')
            return f"(= {left.strip()} {right.strip()})"
            
        return self._convert_expression(constraint)
    
    def _convert_option_check(self, line: str, option: str) -> str:
        """Convert option check to SMT-LIB2 format"""
        match = re.match(r"if is_valid\(Exists\(\[(.*?)\],\s*(.*)\)\)", line)
        if match:
            vars_part, body = match.groups()
            vars_decl = ' '.join(f"({v.split(':')[0].strip()} {v.split(':')[1].strip().replace('_sort', '')})"
                               for v in vars_part.split(','))
            body = self._convert_expression(body)
            return f"""; Option {option}
(push)
(assert (not (exists ({vars_decl}) {body})))
(check-sat)
(pop)"""
        return None
    
    def _convert_expression(self, expr: str) -> str:
        """Convert Z3 expressions to SMT-LIB2 format"""
        # Convert equality
        expr = re.sub(r'([^=])==([^=])', r'\1=\2', expr)
        
        # Convert inequality
        expr = re.sub(r'!=', 'distinct', expr)
        
        # Convert logical operators
        expr = re.sub(r'Or\((.*?)\)', r'(or \1)', expr)
        expr = re.sub(r'And\((.*?)\)', r'(and \1)', expr)
        expr = re.sub(r'Not\((.*?)\)', r'(not \1)', expr)
        
        # Clean up any remaining syntax
        expr = expr.replace('(', ' ( ').replace(')', ' ) ')
        expr = ' '.join(expr.split())
        return expr

def convert_z3_to_smtlib(z3_code: str) -> str:
    """Convert Z3 Python code to SMT-LIB2 format"""
    converter = Z3toSMTLIB()
    return converter.generate_smtlib2(z3_code)

# Example usage
if __name__ == "__main__":
    z3_code = """
from z3 import *

# Declarations
people_sort, (Vladimir, Wendy) = EnumSort('people', ['Vladimir', 'Wendy'])
meals_sort, (breakfast, lunch, dinner, snack) = EnumSort('meals', ['breakfast', 'lunch', 'dinner', 'snack'])
foods_sort, (fish, hot_cakes, macaroni, omelet, poached_eggs) = EnumSort('foods', ['fish', 'hot_cakes', 'macaroni', 'omelet', 'poached_eggs'])
eats = Function('eats', people_sort, meals_sort, foods_sort)

# Constraints
pre_conditions = []
m = Const('m', meals_sort)
pre_conditions.append(ForAll([m], eats(Vladimir, m) != eats(Wendy, m)))
p = Const('p', people_sort)
pre_conditions.append(ForAll([p], Or(eats(p, lunch) == fish, eats(p, lunch) == hot_cakes)))

# Options
m = Const('m', meals_sort)
if is_valid(Exists([m], eats(Vladimir, m) == fish)): print('(A)')
"""
    
    smtlib2_code = convert_z3_to_smtlib(z3_code)
    print(smtlib2_code)