from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any
import re
import json

class TheoryAnalyzer:
    def __init__(self, facts: List[str], rules: List[str]):
        self.initial_facts = set(facts)
        self.rules = rules
        self.fact_graph = defaultdict(list)
        self.derived_steps = {}
        self.all_facts = set(facts)
        self.derivation_rules = {}
        self.used_facts = {}
        
    def parse_rule(self, rule: str) -> Tuple[List[Tuple[bool, str]], str]:
        """Parse a rule into conditions and conclusion.
        Returns a list of tuples (is_positive, condition) and the conclusion."""
        rule = rule.rstrip('.')
        parts = rule.split(' → ')
        
        # Parse conditions
        conditions_str = parts[0].split(' ∧ ') if '∧' in parts[0] else [parts[0]]
        conditions = []
        for cond in conditions_str:
            if cond.startswith('~'):
                conditions.append((False, cond[1:]))  # (is_positive, condition)
            else:
                conditions.append((True, cond))
                
        conclusion = parts[1]
        return conditions, conclusion
    
    def substitute_variable(self, pattern: str, entity: str) -> str:
        """Substitute ?X in a pattern with a specific entity."""
        return pattern.replace('?X', entity)
    
    def get_entities_from_fact(self, fact: str) -> List[str]:
        """Extract entities from a single fact."""
        # Remove leading negation if present
        fact = fact[1:] if fact.startswith('~') else fact
        entities = []
        matches = re.findall(r'[a-zA-Z]+\(([^,]+),([^)]+)\)', fact)
        if matches:
            entities.extend(m for m in matches[0] if not m.startswith('?'))
        return entities
    
    def get_entities(self) -> Set[str]:
        """Extract all entities from the known facts."""
        entities = set()
        for fact in self.all_facts:
            entities.update(self.get_entities_from_fact(fact))
        return entities

    def get_fact_depth(self, fact: str) -> int:
        """Get the depth (number of steps) required to derive a fact."""
        if fact in self.initial_facts:
            return 0
        return self.derived_steps.get(fact, 0)
    
    def rule_applies(self, conditions: List[Tuple[bool, str]], entity: str) -> bool:
        """Check if all conditions of a rule apply for a given entity."""
        for is_positive, cond in conditions:
            substituted_cond = self.substitute_variable(cond, entity)
            # For positive conditions, check if fact exists
            # For negative conditions, check if fact doesn't exist
            fact_exists = substituted_cond in self.all_facts
            if is_positive != fact_exists:
                return False
        return True

    def calculate_derivation_depth(self, conditions: List[Tuple[bool, str]]) -> int:
        """Calculate the depth required for a new derivation based on its conditions."""
        max_depth = 0
        for _, condition in conditions:
            if condition in self.derived_steps:
                depth = self.get_fact_depth(condition)
                max_depth = max(max_depth, depth)
        return max_depth + 1
    
    def analyze(self) -> Dict[str, int]:
        """Analyze the theory and find all derivable facts with their minimum steps."""
        queue = deque((fact, 0) for fact in self.initial_facts)
        entities = self.get_entities()
        
        while queue:
            current_fact, current_step = queue.popleft()
            
            # Try applying each rule
            for rule in self.rules:
                conditions, conclusion = self.parse_rule(rule)
                
                # If the rule has no variables, check it directly
                if '?X' not in rule:
                    if self.rule_applies(conditions, ''):
                        if conclusion not in self.all_facts:
                            depth = self.calculate_derivation_depth(conditions)
                            self.all_facts.add(conclusion)
                            self.derived_steps[conclusion] = depth
                            self.derivation_rules[conclusion] = rule
                            self.used_facts[conclusion] = [cond for _, cond in conditions]
                            queue.append((conclusion, depth))
                            for _, cond in conditions:
                                self.fact_graph[cond].append((conclusion, rule))
                
                # If the rule has variables, try it with each entity
                else:
                    for entity in entities:
                        if self.rule_applies(conditions, entity):
                            new_fact = self.substitute_variable(conclusion, entity)
                            if new_fact not in self.all_facts:
                                substituted_conditions = [
                                    self.substitute_variable(cond, entity) 
                                    for _, cond in conditions
                                ]
                                depth = self.calculate_derivation_depth(conditions)
                                self.all_facts.add(new_fact)
                                self.derived_steps[new_fact] = depth
                                self.derivation_rules[new_fact] = rule
                                self.used_facts[new_fact] = substituted_conditions
                                queue.append((new_fact, depth))
                                for cond in substituted_conditions:
                                    self.fact_graph[cond].append((new_fact, rule))
        
        return self.derived_steps
    
    def get_derivation_path(self, target_fact: str) -> List[Dict[str, Any]]:
        """Get detailed derivation steps for a specific fact."""
        if target_fact not in self.derived_steps:
            return []
        
        derivation_steps = []
        visited = set()
        
        def collect_derivation_info(current_fact: str, step: int) -> None:
            if current_fact in visited or current_fact in self.initial_facts:
                return
                
            visited.add(current_fact)
            
            # Get the facts that were actually used to derive this fact
            used_facts = self.used_facts.get(current_fact, [])
            
            # Get the rule that was used to derive this fact
            rule_used = self.derivation_rules.get(current_fact, "")
            
            # Create derivation step info
            step_info = {
                "step": step,
                "new_fact": current_fact,
                "used_facts": sorted(list(used_facts)),
                "rule_applied": rule_used
            }
            derivation_steps.append(step_info)
            
            # Recurse on facts that were used in the derivation
            for used_fact in used_facts:
                collect_derivation_info(used_fact, step - 1)
        
        collect_derivation_info(target_fact, self.derived_steps[target_fact])
        return sorted(derivation_steps, key=lambda x: x["step"])

# Example usage function
def analyze_theory(theory_data: Dict) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze a theory and return new facts with their detailed derivation paths."""
    analyzer = TheoryAnalyzer(theory_data['facts'], theory_data['rules'])
    derived_steps = analyzer.analyze()
    
    # Organize results
    new_facts = {}
    for fact, steps in derived_steps.items():
        if fact not in theory_data['facts']:  # Only include newly derived facts
            derivation_path = analyzer.get_derivation_path(fact)
            new_facts[fact] = derivation_path
            
    return new_facts

# Example usage
if __name__ == "__main__":
    test_theory = {
        "facts": [
            "is(Alan,Young)",
            "is(Alan,Round)",
            "likes(Alan,Bob)"
        ],
        "rules": [
            "~is(?X,Young) ∧ is(?X,Round) → is(?X,Kind).",
            "likes(?X,Bob) ∧ is(?X,Kind) → is(?X,Short)."
        ]
    }
    
    print("Analyzing test theory...")
    new_facts = analyze_theory(test_theory)
    for fact, derivation in new_facts.items():
        print(f"\nDerivation for: {fact}")
        for step in derivation:
            print(f"\nStep {step['step']}:")
            print(f"New fact: {step['new_fact']}")
            print("Used facts:")
            for used_fact in step['used_facts']:
                print(f"  {used_fact}")
            print(f"Rule applied: {step['rule_applied']}")