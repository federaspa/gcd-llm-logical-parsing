import json
import re
import tempfile
import os
import subprocess
from typing import Tuple, Optional, List, Dict, Any

class Clingo_Program:
    def __init__(self, logic_program: str, dataset_name) -> None:
        """Initialize the Clingo program executor."""
        self.logic_program = logic_program
        self.flag, self.formula_error = self.parse_logic_program(), ""
        self.dataset_name = dataset_name
        
        self.answer_map = {
            'ProntoQA': self.answer_map_prontoqa,
            'ProofWriter': self.answer_map_proofwriter
        }
    
    def parse_logic_program(self) -> bool:
        """Parse the JSON ASP program."""
        try:
            # Load the JSON string
            if isinstance(self.logic_program, str):
                try:
                    program_data = json.loads(self.logic_program)
                except json.JSONDecodeError:
                    program_data = self.logic_program
            else:
                program_data = self.logic_program
            
            # Extract components
            self.Facts = program_data.get('facts', [])
            self.Rules = program_data.get('rules', [])
            self.Query = program_data.get('query', '')
            
            # Validate the program
            return self.validate_program()
        except Exception as e:
            self.formula_error = str(e)
            return False
    
    def validate_program(self) -> bool:
        """Check if the program is valid."""
        if self.Rules is None or self.Facts is None:
            return False
        
        if not self.Facts and not self.Rules:
            return False
        
        return True
    
    def build_program_string(self) -> str:
        """Build a string containing the complete ASP program."""
        program_str = ""
        
        # Add facts
        for fact in self.Facts:
            program_str += f"{fact}\n"
        
        # Add rules
        for rule in self.Rules:
            program_str += f"{rule}\n"
        
        # Add query as a special rule to check if it's derivable
        if self.Query:
            # Remove trailing period if present
            query = self.Query.rstrip('.')
            
            # Add a rule to check if the query is true
            program_str += f"\n% Query verification\n"
            program_str += f"query :- {query}.\n"
            
            # Add a show statement for the query atom
            program_str += f"#show query/0.\n"
        
        return program_str
    
    def execute_program(self) -> Tuple[Optional[str], str]:
        """Execute the ASP program using clingo as a subprocess to avoid segfaults."""
        if not self.flag:
            return None, f"Invalid program: {self.formula_error}"
        
        try:
            # Build the program string
            program_str = self.build_program_string()
            
            # Create a temporary file for the program
            with tempfile.NamedTemporaryFile('w', suffix='.lp', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(program_str)
            
            try:
                # Run clingo as a subprocess
                cmd = ['clingo', tmp_path, '0']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Check for errors
                if result.returncode not in [0, 10, 20, 30]:  # Valid clingo exit codes
                    return None, f"Clingo error: {result.stderr}"
                
                # Parse the result for our query atom
                output = result.stdout
                
                # Check if program is satisfiable
                if "UNSATISFIABLE" in output:
                    query_found = False
                else:
                    # Look for query atom in the answer sets
                    query_found = 'query' in output and not ('query :' in output)
                    
                    # If not found but we have facts matching the query, it's still true
                    if not query_found:
                        query_atom = self.Query.rstrip('.')
                        query_found = query_atom in [fact.rstrip('.') for fact in self.Facts]
                        
                
                answer = self.answer_map[self.dataset_name](query_found)
                
                return answer, ""
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        except Exception as e:
            return None, str(e)
    
    def answer_map_prontoqa(self, result: bool) -> str:
        """Map the result to ProntoQA format."""
        return 'A' if result else 'B'
    
    def answer_map_proofwriter(self, result: Optional[bool]) -> str:
        """Map the result to ProofWriter format."""
        if result is None:
            return 'C'
        return 'A' if result else 'B'
    
    def extract_predicate_info(self, query: str) -> Tuple[str, str, bool]:
        """Extract predicate, subject, and value from a query like 'predicate(subject, value)'."""
        pattern = r'(\w+)\(([^,]+),\s*([^)]+)\)'
        match = re.match(pattern, query)
        if match:
            predicate = match.group(1)
            subject = match.group(2)
            value_str = match.group(3)
            value = (value_str.strip() == 'True')
            return predicate, subject, value
        else:
            raise ValueError(f'Invalid query: {query}')


if __name__ == "__main__":
    # Example JSON input - this should now correctly return "A" (True)
    json_input = """
    {
      "facts": [
        "cold(dog)."
      ],
      "rules": [
      ],
      "query": "cold(dog)."
    }
    """
    
    clingo_program = Clingo_Program(json_input, 'ProofWriter')
    result, error_message = clingo_program.execute_program()
    print(f"Result: {result}")
    if error_message:
        print(f"Error: {error_message}")
        
    # Another example with rules
    json_input2 = """
    {
    "facts": [
        "sees(dog, rabbit).",
        "sees(dog, squirrel).",
        "sees(dog, tiger).",
        "eats(rabbit, dog).",
        "not eats(rabbit, tiger).",
        "not likes(rabbit, tiger).",
        "not sees(squirrel, rabbit).",
        "not eats(tiger, rabbit).",
        "not kind(tiger).",
        "likes(tiger, dog).",
        "sees(tiger, dog)."
    ],
    "rules": [
        "likes(X, rabbit) :- cold(X).",
        "likes(X, rabbit) :- eats(X, tiger), nice(X).",
        "likes(squirrel, rabbit) :- likes(X, squirrel).",
        "sees(X, tiger) :- likes(X, rabbit), kind(rabbit).",
        "young(tiger) :- likes(X, tiger).",
        "likes(X, tiger) :- young(X), eats(X, rabbit).",
        "cold(rabbit) :- sees(X, rabbit).",
        "likes(X, squirrel) :- likes(X, rabbit).",
        "cold(squirrel) :- likes(X, squirrel)."
    ],
    "query": "hot(rabbit)"
    }
    """
    
    clingo_program2 = Clingo_Program(json_input2, 'ProofWriter')
    result2, error_message2 = clingo_program2.execute_program()
    print(f"Result 2: {result2}")
    if error_message2:
        print(f"Error: {error_message2}")