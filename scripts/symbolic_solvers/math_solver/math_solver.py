import sympy as sp
from typing import Tuple, Dict

class SymPy_Program:
    def __init__(self, logic_program: str, dataset_name='GSM8K_symbolic') -> None:
        self.logic_program = logic_program
        self.flag, self.formula_error = self.parse_logic_program()
        self.dataset_name = dataset_name

    def parse_logic_program(self) -> Tuple[bool, str | None]:
        try:
            premises = self.logic_program.get("data", [])
            question = self.logic_program.get("question", "")
            
            # Define a dictionary to hold the variables and their values
            self.variables = {}
            
            for premise in premises:
                if "=" in premise:
                    key, value_str = premise.split("=")
                    try:
                        value_expr = sp.sympify(value_str.strip())
                        self.variables[key.strip()] = value_expr
                    except Exception as e:
                        return False, f"Error parsing premise '{premise}': {str(e)}"
                else:
                    return False, f"Invalid premise format: {premise}"
            
            try:
                self.simp_logic_program = sp.sympify(question)
            except Exception as e:
                return False, f"Error parsing question '{question}': {str(e)}"
            
            return True, None
        except Exception as e:
            return False, str(e)

    def execute_program(self) -> Tuple[str | None, str]:
        if not self.flag:
            return None, self.formula_error
        
        try:
            # Substitute the variables in the expression
            result = self.simp_logic_program.subs(self.variables)
            
            # Evaluate the result to a numerical value if possible
            evaluated_result = sp.N(result)
            
            return str(evaluated_result), ""
        except Exception as e:
            return None, f"Error executing program: {str(e)}"
        
    @staticmethod
    def answer_mapping(answer:str) -> str:
        try:
            return float(answer)
        except:
            return answer

if __name__ == "__main__":
    # Example usage:
    problem_str = """{
        "premises": [
            "whale_length = 380 * 12",
            "remora_length = 57",
            "num_remoras = 8"
        ],
        "question": "((num_remoras * remora_length) / whale_length) * 100"
    }"""

    symPy_program = SymPy_Program(problem_str)
    if symPy_program.flag:
        answer, error = symPy_program.execute_program()
        if not error:
            print(f"Answer: {answer}")
        else:
            print(f"Error: {error}")
    else:
        print(f"Parsing Error: {symPy_program.formula_error}")