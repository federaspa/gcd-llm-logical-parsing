import subprocess
import tempfile
import re
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional
import os
import json
from theory_generator import LogicalTheoryGenerator

@dataclass
class ProofStep:
    conclusion: str
    depth: int
    premises: List[str]

@dataclass
class Theory:
    facts: List[str]
    rules: List[str]

class VampireReasoner:
    def __init__(self, vampire_path: str = "/home/fraspant/vampire"):
        """Initialize the reasoner with path to Vampire executable."""
        self.vampire_path = vampire_path
        # Verify Vampire is available
        try:
            subprocess.run([vampire_path, "--version"], capture_output=True)
        except FileNotFoundError:
            raise RuntimeError("Vampire theorem prover not found. Please ensure it's installed and in PATH")

    def theory_to_tptp(self, theory: Theory) -> str:
        """Convert a theory to TPTP format that Vampire can process."""
        tptp_lines = []
        
        # Convert facts to TPTP format
        for i, fact in enumerate(theory.facts):
            # Convert is(Entity,Attribute) to TPTP
            # Example: is(Alan,Young) becomes fof(fact1, axiom, is(alan,young)).
            fact = fact.replace("(", "(").replace(")", ")")
            tptp_lines.append(f"fof(fact{i}, axiom, {fact.lower()}).")

        # Convert rules to TPTP format
        for i, rule in enumerate(theory.rules):
            # Parse the rule into conditions and conclusion
            rule = rule.strip(".")  # Remove trailing period
            parts = rule.split("→")
            conditions = parts[0].strip()
            conclusion = parts[1].strip()
            
            # Handle conditions with AND
            if "∧" in conditions:
                conditions = conditions.replace("∧", "&")
            
            # Convert to TPTP implication format
            rule_tptp = f"fof(rule{i}, axiom, ![X] : ({conditions} => {conclusion}))."
            rule_tptp = rule_tptp.lower().replace("?x", "X").replace("![x]", "![X]")
            tptp_lines.append(rule_tptp)
            
        return "\n".join(tptp_lines)

    def run_vampire(self, tptp_input: str, time_limit: int = 10) -> str:
        """Run Vampire on the given TPTP input with specified time limit."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.p', delete=False) as f:
            f.write(tptp_input)
            input_file = f.name

        try:
            # Run Vampire with options to output proof steps
            result = subprocess.run(
                [self.vampire_path, 
                 "--proof", "tptp",
                #  "--proof_extra", "full",
                 "--mode", "casc",
                 "--show_new", "on",
                 "--schedule", "induction",
                 "--time_limit", str(time_limit),
                 input_file],
                capture_output=True,
                text=True
            )
            return result.stdout
        finally:
            os.unlink(input_file)

    def extract_conclusions_by_depth(self, theory: Theory, max_depth: int) -> Dict[int, Set[str]]:
        """Extract all conclusions up to specified depth from the theory."""
        # Convert theory to TPTP
        tptp_input = self.theory_to_tptp(theory)
        
        # print(tptp_input)
        
        # Run Vampire
        vampire_output = self.run_vampire(tptp_input)
        
        # Parse proof steps and track depths
        conclusions_by_depth: Dict[int, Set[str]] = {d: set() for d in range(max_depth + 1)}
        
        # Add initial facts at depth 0
        for fact in theory.facts:
            conclusions_by_depth[0].add(fact)
            
            
        print(vampire_output)
            
        # Process Vampire output to extract proof steps and their depths
        proof_steps = self._parse_vampire_proof(vampire_output)
        
        # Build dependency graph and compute depths
        for step in proof_steps:
            # Skip steps beyond our max depth
            if step.depth > max_depth:
                continue
            conclusions_by_depth[step.depth].add(step.conclusion)
        
        return conclusions_by_depth

    def _parse_vampire_proof(self, vampire_output: str) -> List[ProofStep]:
        """Parse Vampire's proof output to extract steps and their depths."""
        proof_steps = []
        current_depth = 0
        
        # Regular expressions for matching different parts of Vampire output
        step_pattern = re.compile(r'\[(\d+)\] (.+) \[(.+)\]')
        
        for line in vampire_output.split('\n'):
            if 'SZS output start Proof' in line:
                current_depth = 0
            elif match := step_pattern.search(line):
                step_num, conclusion, premises = match.groups()
                
                # Parse premises to determine depth
                premise_nums = [int(p) for p in premises.split(',') if p.isdigit()]
                if premise_nums:
                    current_depth = max(p for p in premise_nums) + 1
                
                proof_steps.append(ProofStep(
                    conclusion=conclusion.strip(),
                    depth=current_depth,
                    premises=premises.split(',')
                ))
        
        return proof_steps

# Example usage
if __name__ == "__main__":
    # # Load theory from JSON file
    # with open('out.json', 'r') as f:
    #     theory_data = json.load(f)

    # # Create a Theory object from the loaded data
    # theory = Theory(
    #     facts=theory_data.get('facts', []),
    #     rules=theory_data.get('rules', [])
    # )
    # # Create a simple theory
    theory1 = Theory(
        facts=[
            "is(Alan,Young)",
            "is(Alan,Round)",
            "likes(Alan,Bob)"
        ],
        rules=[
            "is(?X,Young) ∧ is(?X,Round) → is(?X,Kind).",
            "likes(?X,Bob) ∧ is(?X,Kind) → is(?X,Short).",
     
    ]
    )
    
    # Initialize reasoner
    reasoner = VampireReasoner()
    
    # Get conclusions up to depth 2
    conclusions = reasoner.extract_conclusions_by_depth(theory1, max_depth=20)
    
    # print('#'*50)
    # # Print conclusions by depth
    # for depth, facts in conclusions.items():
    #     print(f"\nDepth {depth}:")
    #     for fact in facts:
    #         print(f"  {fact}")
