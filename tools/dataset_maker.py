import random
from typing import List, Dict
import json
from utils.symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program

class ValidatingLogicGenerator:
    def __init__(self):
        self.names = ["eli", "patricia", "broderick", "paul", "miles", "ronald", "olive"]
        self.predicates = ["Scared", "Poised", "Soft", "Jittery", "Southern", "Civil"]
        
    def generate_problem(self, reasoning_depth: int) -> Dict:
        """Generate a FOL problem with specified reasoning depth"""
        # Generate target fact and desired answer
        pred = random.choice(self.predicates)
        name = random.choice(self.names)
        is_negative = random.choice([True, False])
        target_fact = f"{'¬' if is_negative else ''}{pred}({name})"
        desired_answer = random.choice(["A", "B"])  # A=true, B=false
        
        # Generate context
        context_fol = []
        
        # Add basic facts
        for _ in range(random.randint(8, 12)):
            p = random.choice(self.predicates)
            n = random.choice(self.names)
            neg = random.choice([True, False])
            fact = f"{'¬' if neg else ''}{p}({n})"
            context_fol.append(fact)
            
        # Add rules with increasing complexity based on depth
        for d in range(reasoning_depth):
            num_rules = random.randint(2, 4)
            for _ in range(num_rules):
                # Generate quantified rules
                quantifier = random.choice(["∀", "∃"])
                preds = random.sample(self.predicates, 2)
                connective = random.choice(["→", "↔"])
                
                if d == 0:
                    # Simple rules for depth 1
                    rule = f"({quantifier}x ({preds[0]}(x))) {connective} ({quantifier}x ({preds[1]}(x)))"
                else:
                    # More complex rules for higher depths
                    left = f"({quantifier}x ({preds[0]}(x) ∧ {preds[1]}(x)))"
                    right = f"({random.choice(self.predicates)}({random.choice(self.names)}))"
                    rule = f"{left} {connective} {right}"
                
                context_fol.append(rule)
        
        return {
            "id": 0,
            "story_id": 0,
            "context_fol": context_fol,
            "question_fol": target_fact,
            "answer": desired_answer,
            "logic_predicates": [f"{pred}(x)" for pred in self.predicates]
        }

    def generate_and_validate_problem(self, reasoning_depth: int, max_attempts: int = 100) -> Dict:
        """Generate a valid FOL problem and verify it with Prover9"""
        for attempt in range(max_attempts):
            problem = self.generate_problem(reasoning_depth)
            
            # Prepare problem for Prover9
            prover9_input = {
                "fol_rules": problem["context_fol"],
                "fol_conc": problem["question_fol"]
            }
            
            # Create Prover9 program
            prover = FOL_Prover9_Program(prover9_input)
            
            if not prover.flag:
                continue  # Skip if parsing failed
                
            # Execute the program
            answer, error = prover.execute_program()
            if error:
                continue
                
            # Check if we got the desired answer
            try:
                mapped_answer = prover.answer_mapping(answer)
                if mapped_answer == problem["answer"]:
                    return problem
            except:
                continue
                
        raise ValueError("Could not generate valid problem within maximum attempts")

# Example usage
generator = ValidatingLogicGenerator()
try:
    valid_problem = generator.generate_and_validate_problem(reasoning_depth=15)
    print(json.dumps(valid_problem, indent=2, ensure_ascii=False))
except ValueError as e:
    print(f"Error: {e}")