import json
from pyswip import Prolog
import tempfile
import os

def create_prolog_program(data, temp_file):
    """Creates a Prolog program file from the provided data structure."""
    with open(temp_file, 'w') as f:
        # Write all predicates as dynamic
        for predicate in data['predicates']:
            f.write(f':- dynamic {predicate}/1.\n')
        
        # Write all facts
        for fact in data['facts']:
            f.write(f'{fact}.\n')
        
        # Write all rules
        for rule in data['rules']:
            # Replace Python-style backslash with Prolog-style
            # rule = rule.replace('\\+', 'not')
            f.write(f'{rule}.\n')

def run_prolog_query(program_data):
    """
    Runs the Prolog program with the given query and returns the result.
    
    Args:
        program_data (dict): Dictionary containing predicates, facts, rules, and query
    
    Returns:
        bool: Result of the query
    """
    # Create a temporary file for the Prolog program
    # Explicitly use forward slashes for the temp directory
    # temp_dir = tempfile.gettempdir().replace('\\', '/')
    # fd, temp_path = tempfile.mkstemp(suffix='.pl', dir=temp_dir)
    # os.close(fd)
    # temp_filename = temp_path.replace('\\', '/')
    temp_filename = 'here'
    
    # try:
    # Create the Prolog program file
    create_prolog_program(program_data, temp_filename)
    
    # Initialize Prolog
    prolog = Prolog()
    
    # Consult the program file using forward slashes
    prolog.consult(temp_filename)
    
    # Run the query
    query = program_data['query']
    result = bool(list(prolog.query(query)))
    
    return result
    
    # finally:
    #     # Clean up the temporary file
    #     os.unlink(temp_filename)

def main():
    # Your Prolog program data
    program_data = {
        "predicates": ["dumpus", "impuses", "jompus", "numpus", "rompus", 
                      "tumpuses", "vumpuses", "wumpus", "wumpuses", "yumpus", "shy"],
        "facts": ["tumpuses(alex)"],
        "rules": [
            "jompus(X) :- fruity(X)",
            "jompus(X) :- wumpus(X)",
            "wumpus(X) :- \\+transparent(X)",
            "wumpuses(X) :- tumpuses(X)",
            "tumpuses(X) :- mean(X)",
            "tumpuses(X) :- vumpuses(X)",
            "vumpuses(X) :- cold(X)",
            "vumpuses(X) :- yumpus(X)",
            "yumpus(X) :- orange(X)",
            "yumpus(X) :- numpus(X)",
            "numpus(X) :- dull(X)",
            "numpus(X) :- dumpus(X)",
            "dumpus(X) :- \\+shy(X)",
            "impuses(X) :- shy(X)",
            "dumpus(X) :- rompus(X)",
            "rompus(X) :- liquid(X)",
            "rompus(X) :- zumpus(X)"
        ],
        "query": "\\+shy(alex)"
    }
    
    # Run the query and get the result
    result = run_prolog_query(program_data)
    print(f"Query result: {result}")

if __name__ == "__main__":
    main()