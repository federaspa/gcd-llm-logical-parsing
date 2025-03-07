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
            'ProofWriter': self.answer_map_proofwriter,
            'LSAT': self.answer_map_lsat
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
            
            # Extract options for LSAT-type problems
            self.Options = program_data.get('options', [])
            self.Label = program_data.get('label', '')  # Ground truth label if available
            
            # Determine the problem type based on presence of options
            self.is_option_problem = len(self.Options) > 0
            
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
        
        if self.is_option_problem and not self.Options:
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
        
        # For option-based problems, add options
        if self.is_option_problem:
            for option in self.Options:
                program_str += f"{option}\n"
            
            # Add the query if provided (usually the answer rule)
            if self.Query:
                program_str += f"{self.Query}\n"
                program_str += "#show answer/1.\n"
        # For yes/no query problems    
        elif query:
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
        """Execute the ASP program based on the problem type."""
        if not self.flag:
            return None, f"Invalid program: {self.formula_error}"
        
        try:
            # Handle option-based problems differently
            if self.is_option_problem:
                return self._execute_option_program()
            else:
                return self._execute_query_program()
                
        except Exception as e:
            return None, str(e)
    
    def _execute_option_program(self) -> Tuple[Optional[str], str]:
        """Execute an option-based ASP program to find which option is valid."""
        # Build the program string with options
        program_str = self.build_program_string()
        
        # Create a temporary file for the program
        with tempfile.NamedTemporaryFile('w', suffix='.lp', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(program_str)
            print(program_str)
        
        try:
            # Run clingo as a subprocess
            cmd = ['clingo', tmp_path, '0']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check for errors
            if result.returncode not in [0, 10, 20, 30]:  # Valid clingo exit codes
                return None, f"Clingo error: {result.stderr}"
            
            # Parse the result to find the valid option
            output = result.stdout
            
            # Check if program is unsatisfiable
            if "UNSATISFIABLE" in output:
                return None, "No valid option found (UNSATISFIABLE)"
            
            # Extract the answer atom from the output
            # Pattern matches answer(X) where X is the option letter
            answer_match = re.search(r'answer\((.*?)\)', output)
            if answer_match:
                option_letter = answer_match.group(1)
                # Return the option letter through the answer map
                return self.answer_map_lsat(option_letter), ""
            else:
                return None, "No answer found in the output"
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def _execute_query_program(self) -> Tuple[Optional[str], str]:
        """Execute a query-based ASP program (original method)."""
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
    
    def answer_map_lsat(self, option_letter: str) -> str:
        """Map the option letter to a standardized format."""
        # Ensure the option letter is uppercase
        return option_letter.upper()
    
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
    
    def check_solution(self) -> bool:
        """Check if the computed solution matches the ground truth label."""
        solution, error = self.execute_program()
        if error:
            return False
        return solution == self.Label


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
        
    json_input3 = """
{
  "facts": [
    "tumpus(alex)."
  ],
  "rules": [
    "fruity(X) :- jompus(X).",
    "wumpus(X) :- jompus(X).",
    "not transparent(X) :- wumpus(X).",
    "tumpus(X) :- wumpus(X).",
    "mean(X) :- tumpus(X).",
    "vumpus(X) :- tumpus(X).",
    "cold(X) :- vumpus(X).",
    "yumpus(X) :- vumpus(X).",
    "orange(X) :- yumpus(X).",
    "numpus(X) :- yumpus(X).",
    "dull(X) :- numpus(X).",
    "dumpus(X) :- numpus(X).",
    "not shy(X) :- dumpus(X).",
    "shy(X) :- impus(X).",
    "rompus(X) :- dumpus(X).",
    "liquid(X) :- rompus(X).",
    "zumpus(X) :- rompus(X)."
  ],
  "query": "not shy(alex)"
}
    """
    
    clingo_program3 = Clingo_Program(json_input3, 'ProofWriter')
    result3, error_message3 = clingo_program3.execute_program()
    print(f"Result 3: {result3}")
    if error_message3:
        print(f"Error: {error_message3}")


    json_input4 = """
{
  "facts": [
  "big(dave).",
  "red(dave).",
  "smart(erin).",
  "kind(fiona).",
  "smart(fiona).",
  "rough(gary).",
  "white(gary)."
  ],
  "rules": [
    "white(X) :- young(X).",
    "big(X) :- kind(X), white(X).",
    "young(X) :- kind(X).",
    "red(fiona) :- young(fiona), rough(fiona).",
    "rough(X) :- big(X).",
    "red(X) :- rough(X), white(X).",
    "red(X) :- kind(X), not big(X)."
  ],
  "query": "smart(erin)"
}
    """

    clingo_program4 = Clingo_Program(json_input4, 'ProofWriter')
    result4, error_message4 = clingo_program4.execute_program()
    print(f"Result 4: {result4}")
    if error_message4:
        print(f"Error: {error_message4}")
        
        
    json_input5 = """
{
  "facts": [
    "person(klosnik).",
    "person(londi).",
    "person(manley).",
    "person(neri).",
    "person(osata).",
    "person(poirier).",
    "seat(1).",
    "seat(2).",
    "seat(3).",
    "seat(4).",
    "seat(5).",
    "seat(6).",
    "next(1, 2).",
    "next(2, 3).",
    "next(3, 4).",
    "next(4, 5).",
    "next(5, 6).",
    "next(6, 1)."
  ],
  "rules": [
    "1 { sits(P, S) : seat(S) } 1 :- person(P).",
    "1 { sits(P, S) : person(P) } 1 :- seat(S).",
    "adjacent(X, Y) :- next(X, Y).",
    "adjacent(X, Y) :- next(Y, X).",
    "adjacent(P1, P2) :- sits(P1, S1), sits(P2, S2), adjacent(S1, S2).",
    ":- not adjacent(poirier, neri).",
    ":- not adjacent(londi, manley), not adjacent(londi, neri).",
    ":- adjacent(klosnik, manley).",
    ":- adjacent(osata, poirier), adjacent(osata, manley)."
  ],
  "options": [
    "option(a) :- sits(klosnik, 1), sits(poirier, 2), sits(neri, 3), sits(manley, 4), sits(osata, 5), sits(londi, 6).",
    "option(b) :- sits(klosnik, 1), sits(londi, 2), sits(manley, 3), sits(poirier, 4), sits(neri, 5), sits(osata, 6).",
    "option(c) :- sits(klosnik, 1), sits(londi, 2), sits(manley, 3), sits(osata, 4), sits(poirier, 5), sits(neri, 6).",
    "option(d) :- sits(klosnik, 1), sits(osata, 2), sits(poirier, 3), sits(neri, 4), sits(londi, 5), sits(manley, 6).",
    "option(e) :- sits(klosnik, 1), sits(neri, 2), sits(londi, 3), sits(osata, 4), sits(manley, 5), sits(poirier, 6)."
  ],
  "query": "answer(X) :- option(X), not invalid(X).",
  "label": "B"
}
    """

    clingo_program5 = Clingo_Program(json_input5, 'LSAT')
    result5, error_message5 = clingo_program5.execute_program()
    print(f"Result 5: {result5}")
    if error_message5:
        print(f"Error: {error_message5}")

    json_input6 = """
{
  "facts": [
    "office(1).",
    "office(2).",
    "office(3).",
    "office(4).",
    "year(1987).",
    "year(1988).",
    "year(1989).",
    "computer_bought(1, 1988).",
    "printer_bought(3, 1988)."
  ],
  "rules": [
    "1 { computer_bought(O, Y) : year(Y) } 1 :- office(O).",
    "1 { printer_bought(O, Y) : year(Y) } 1 :- office(O).",
    ":- computer_bought(O, Y1), printer_bought(O, Y2), Y1 > Y2.",
    ":- computer_bought(2, Y1), printer_bought(1, Y2), Y1 != Y2.",
    ":- computer_bought(3, Y1), printer_bought(4, Y2), Y1 != Y2.",
    ":- computer_bought(2, Y), computer_bought(3, Y).",
    ":- computer_bought(3, Y1), printer_bought(3, Y2), Y1 >= Y2.",
    "premise :- computer_bought(3, Y1), printer_bought(3, Y2), Y1 < Y2."
  ],
  "options": [
    "option(a) :- premise, computer_bought(2, 1987).",
    "option(b) :- premise, computer_bought(2, 1988).",
    "option(c) :- premise, computer_bought(4, 1988).",
    "option(d) :- premise, printer_bought(4, 1988).",
    "option(e) :- premise, printer_bought(4, 1989)."
  ],
  "query": "answer(X) :- option(X), not invalid(X).",
  "label": "B"
}
    """
    
    clingo_program6 = Clingo_Program(json_input6, 'LSAT')
    result6, error_message6 = clingo_program6.execute_program()
    print(f"Result 6: {result6}")
    if error_message6:
        print(f"Error: {error_message6}")

    json_input7 = """
{
  "facts": [
    "worker(brandt).",
    "worker(calva).",
    "worker(duvall).",
    "worker(eberle).",
    "worker(fu).",
    "worker(garcia).",
    "worker(haga).",
    "worker(irving).",
    "worker(jessup).",
    "day(1..9)."
  ],
  "rules": [
    "1 { hired(W, D) : day(D) } 1 :- worker(W).",
    "same_day(fu, irving).",
    "same_day(calva, garcia).",
    "same_day(X, Y) :- hired(X, D), hired(Y, D), X != Y.",
    ":- same_day(X, Y), same_day(X, Z), Y != Z.",
    ":- same_day(fu, irving), same_day(fu, W), W != irving, W != fu.",
    ":- same_day(calva, garcia), same_day(calva, W), W != garcia, W != calva.",
    "before(X, Y) :- hired(X, D1), hired(Y, D2), D1 < D2.",
    ":- not before(eberle, brandt).",
    ":- not before(haga, duvall).",
    ":- not before(irving, duvall).",
    ":- not before(duvall, eberle).",
    ":- not before(jessup, garcia).",
    ":- not before(brandt, garcia).",
    ":- not before(brandt, jessup)."
  ],
  "options": [
    "option(a) :- hired(eberle, D1), hired(jessup, D2), D1 > D2, D1 = 9, D2 = 8.",
    "option(b) :- hired(brandt, D1), hired(garcia, D2), D1 > D2, D1 = 9, D2 = 8.",
    "option(c) :- hired(brandt, D1), hired(calva, D2), D1 > D2, D1 = 9, D2 = 8.",
    "option(d) :- hired(garcia, D1), hired(calva, D2), D1 = D2, D1 = 9.",
    "option(e) :- hired(jessup, D1), hired(brandt, D2), D1 > D2, D1 = 9, D2 = 8."
  ],
  "query": "answer(X) :- option(X), not invalid(X).",
  "label": "D"
}
    """
    
    clingo_program7 = Clingo_Program(json_input7, 'LSAT')
    result7, error_message7 = clingo_program7.execute_program()
    print(f"Result 7: {result7}")
    if error_message7:
        print(f"Error: {error_message7}")

    json_input8 = """
{
"facts": [
"parent(f).",
"parent(g).",
"parent(h).",
"student(k).",
"student(l).",
"student(m).",
"teacher(u).",
"teacher(w).",
"teacher(x).",
"teacher(z).",
"person(P) :- parent(P).",
"person(P) :- student(P).",
"person(P) :- teacher(P)."
],

"rules": [
"{ selected(P) : person(P) } = 5.",
"1 { selected(S) : student(S) } 1.",
":- selected(f), selected(h).",
":- selected(m), selected(z).",
":- selected(u), selected(w).",
":- selected(f), not selected(z).",
":- selected(w), not selected(h).",
"cannot_both(A,B) :- person(A), person(B), A != B, conflict(A,B).",
"conflict(f,h).",
"conflict(m,z).",
"conflict(u,w).",
"conflict(f,m) :- not selected(z).",
"conflict(m,f) :- not selected(z)."
],

"options": [
"option(a) :- cannot_both(f,g).",
"option(b) :- cannot_both(f,m).",
"option(c) :- cannot_both(g,k).",
"option(d) :- cannot_both(h,l).",
"option(e) :- cannot_both(m,u)."
],

"query": "answer(X) :- option(X), not invalid(X)."
}
    """
    
    clingo_program8 = Clingo_Program(json_input8, 'LSAT')
    result8, error_message8 = clingo_program8.execute_program()
    print(f"Result 8: {result8}")
    if error_message8:
        print(f"Error: {error_message8}")