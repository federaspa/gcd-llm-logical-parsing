import json
import re
import tempfile
import os
import subprocess
from typing import Tuple, Optional, List, Dict, Any

class Clingo_Program:
    def __init__(self, logic_program: str, dataset_name='ProntoQA') -> None:
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
    
    def build_program_string(self, query=None, negated=False) -> str:
        """Build a string containing the complete ASP program."""
        program_str = ""
        
        # Add facts
        for fact in self.Facts:
            program_str += f"{fact}\n"
        
        # Add rules
        for rule in self.Rules:
            program_str += f"{rule}\n"
        
        # Add query as a special rule to check if it's derivable
        if query:
            # Remove trailing period if present
            clean_query = query.rstrip('.')
            
            if negated:
                # For negated query, we need to check if the negation can be derived
                # In ASP, we can use constraints and auxiliary atoms to check negation
                program_str += f"\n% Negated Query verification\n"
                program_str += f"neg_query :- not {clean_query}.\n"
                program_str += f"#show neg_query/0.\n"
            else:
                # Add a rule to check if the query is true
                program_str += f"\n% Query verification\n"
                program_str += f"query :- {clean_query}.\n"
                program_str += f"#show query/0.\n"
        
        return program_str
    
    def execute_program(self) -> Tuple[Optional[str], str]:
        """Execute the ASP program using clingo as a subprocess to avoid segfaults."""
        if not self.flag:
            return None, f"Invalid program: {self.formula_error}"
        
        try:
            # First try to prove the query
            query_result = self._run_query(self.Query, False)
            
            # If query is provably true, return True
            if query_result:
                return self.answer_map[self.dataset_name](True), ""
            
            # Now try to prove the negation of the query
            negated_result = self._run_query(self.Query, True)
            
            # If negation is provably true, query is False
            # If neither query nor its negation can be proved, result is Unknown
            is_false = negated_result
            is_unknown = not query_result and not negated_result
            
            if is_unknown:
                return self.answer_map[self.dataset_name](None), ""
            else:
                return self.answer_map[self.dataset_name](not is_false), ""
                
        except Exception as e:
            return None, str(e)
    
    def _run_query(self, query: str, negated: bool = False) -> bool:
        """Run a query and determine if it's provable."""
        # Build the program string
        program_str = self.build_program_string(query, negated)
        
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
                return False
            
            # Parse the result for our query atom
            output = result.stdout
            
            # Check if program is satisfiable
            if "UNSATISFIABLE" in output:
                return False
            
            # Look for the appropriate query atom in the answer sets
            query_atom = "query" if not negated else "neg_query"
            query_found = query_atom in output and not (f"{query_atom} :" in output)
            
            # If not found directly but we have facts matching the query, it's still true
            if not query_found and not negated:
                query_atom = query.rstrip('.')
                query_found = query_atom in [fact.rstrip('.') for fact in self.Facts]
            
            return query_found
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
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
    # Example JSON input - this should correctly return "A" (True)
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
        
    # Example that should return "B" (False)
    json_input2 = """
    {
      "facts": [
        "not cold(dog)."
      ],
      "rules": [
      ],
      "query": "cold(dog)."
    }
    """
    
    clingo_program2 = Clingo_Program(json_input2, 'ProofWriter')
    result2, error_message2 = clingo_program2.execute_program()
    print(f"Result 2: {result2}")
    if error_message2:
        print(f"Error: {error_message2}")
    
    # Example that should return "C" (Unknown)
    json_input3 = """
    {
      "facts": [
        "cold(dog)."
      ],
      "rules": [
        "hot(X) :- not cold(X)."
      ],
      "query": "hot(cat)."
    }
    """
    
    clingo_program3 = Clingo_Program(json_input3, 'ProofWriter')
    result3, error_message3 = clingo_program3.execute_program()
    print(f"Result 3: {result3}")
    if error_message3:
        print(f"Error: {error_message3}")