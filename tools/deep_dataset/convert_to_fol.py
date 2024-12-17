import re
from typing import List, Dict

class FOLConverter:
    def __init__(self):
        # Regular expressions for parsing
        self.is_pattern = re.compile(r'is\(([^,]+),([^)]+)\)')
        self.pred_pattern = re.compile(r'([a-zA-Z]+)\(([^,]+),([^)]+)\)')
        self.var_pattern = re.compile(r'\?X')

    def convert_atom(self, atom: str) -> str:
        """Convert a single atomic formula to FOL notation."""
        negated = atom.startswith('~')
        atom = atom.replace('~', '')
        # Check for is() predicate
        is_match = self.is_pattern.match(atom)
        if is_match:
            subject, predicate = is_match.groups()
            # Convert ?X to x for variables
            subject = 'x' if subject == '?X' else subject.lower()
            
            if negated:
                return f"¬{predicate}({subject})"
            return f"{predicate}({subject})"
        
        # Check for other predicates
        pred_match = self.pred_pattern.match(atom)
        if pred_match:
            predicate, arg1, arg2 = pred_match.groups()
            # Convert ?X to x for variables
            arg1 = 'x' if arg1 == '?X' else arg1.lower()
            arg2 = 'x' if arg2 == '?X' else arg2.lower()
            # Capitalize predicate
            predicate = predicate.capitalize()
            if negated:
                return f"¬{predicate}({arg1}, {arg2})"
            return f"{predicate}({arg1}, {arg2})"
                
        if negated:
            return f'¬{atom}'
        return atom

    def convert_formula(self, formula: str) -> str:
        """Convert a complete formula to FOL notation."""
        # Remove trailing period
        formula = formula.rstrip('.')
        universal_quantified = False
        
        # Add universal quantifier if formula starts with a variable
        if '?X' in formula:
            universal_quantified = True
        
        # Split formula into parts by → and ∧
        parts = re.split(r'(→|∧)', formula)
        parts = [p.strip() for p in parts]
        
            
        # Convert each atomic formula
        for i in range(0, len(parts), 2):
            parts[i] = self.convert_atom(parts[i])
            
        # Rejoin the formula
        formula = ' '.join(parts)
        
        if universal_quantified:
            formula = f"∀x ({formula})"
            # Replace all remaining ?X with x
            formula = formula.replace('?X', 'x')
        
        return formula

def test_converter():
    converter = FOLConverter()
    
    # Test cases
    test_cases = [
        ("is(Alan,Young)", "Young(alan)"),
        ("~is(Alan,Young)", "¬Young(alan)"),
        ("is(?X,Big) → is(?X,Strong)", "∀x (Big(x) → Strong(x))"),
        ("~is(?X,Young) ∧ is(?X,Round) → is(?X,Kind)", "∀x (¬Young(x) ∧ Round(x) → Kind(x))"),
        ("likes(Alan,Mouse)", "Likes(alan, mouse)"),
        ("likes(?X,Dog) → loves(?X,Dog)", "∀x (Likes(x, dog) → Loves(x, dog))"),
        ("chases(Cat,Mouse) ∧ is(Mouse,Small)", "Chases(cat, mouse) ∧ Small(mouse)"),
        ("~chases(Cat,Mouse) ∧ ~is(Mouse,Small)", "¬Chases(cat, mouse) ∧ ¬Small(mouse)")
    ]
    
    print("Testing FOL Converter:")
    print("=====================")
    for input_formula, expected_output in test_cases:
        output = converter.convert_formula(input_formula)
        print(f"\nInput:    {input_formula}")
        print(f"Output:   {output}")
        print(f"Expected: {expected_output}")
        print(f"{'✓' if output == expected_output else '✗'}")

if __name__ == "__main__":
    test_converter()