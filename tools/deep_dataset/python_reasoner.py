from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import re

class TheoryAnalyzer:
    def __init__(self, facts: List[str], rules: List[str]):
        self.initial_facts = set(facts)
        self.rules = rules
        self.fact_graph = defaultdict(list)  # Maps facts to the facts they helped derive
        self.derived_steps = {}  # Maps derived facts to minimum steps needed
        self.all_facts = set(facts)  # All known facts including derived ones
        
    def parse_rule(self, rule: str) -> Tuple[List[str], str]:
        """Parse a rule into conditions and conclusion."""
        # Remove the trailing period
        rule = rule.rstrip('.')
        # Split into conditions and conclusion
        parts = rule.split(' → ')
        conditions = parts[0].split(' ∧ ') if '∧' in parts[0] else [parts[0]]
        conclusion = parts[1]
        return conditions, conclusion
    
    def substitute_variable(self, pattern: str, entity: str) -> str:
        """Substitute ?X in a pattern with a specific entity."""
        return pattern.replace('?X', entity)
    
    def get_entities(self) -> Set[str]:
        """Extract all entities from the known facts."""
        entities = set()
        for fact in self.all_facts:
            match = re.match(r'[a-zA-Z]+\(([^,]+),', fact)
            if match:
                entities.add(match.group(1))
        return entities
    
    def rule_applies(self, conditions: List[str], entity: str) -> bool:
        """Check if all conditions of a rule apply for a given entity."""
        substituted_conditions = [self.substitute_variable(cond, entity) for cond in conditions]
        return all(cond in self.all_facts for cond in substituted_conditions)
    
    def analyze(self) -> Dict[str, int]:
        """Analyze the theory and find all derivable facts with their minimum steps."""
        queue = deque((fact, 0) for fact in self.initial_facts)
        entities = self.get_entities()
        
        while queue:
            current_fact, steps = queue.popleft()
            
            # Try applying each rule
            for rule in self.rules:
                conditions, conclusion = self.parse_rule(rule)
                
                # If the rule has no variables, check it directly
                if '?X' not in rule:
                    if all(cond in self.all_facts for cond in conditions):
                        if conclusion not in self.all_facts:
                            self.all_facts.add(conclusion)
                            self.derived_steps[conclusion] = steps + 1
                            queue.append((conclusion, steps + 1))
                            # Track which facts helped derive this one
                            for cond in conditions:
                                self.fact_graph[cond].append(conclusion)
                
                # If the rule has variables, try it with each entity
                else:
                    for entity in entities:
                        if self.rule_applies(conditions, entity):
                            new_fact = self.substitute_variable(conclusion, entity)
                            if new_fact not in self.all_facts:
                                self.all_facts.add(new_fact)
                                self.derived_steps[new_fact] = steps + 1
                                queue.append((new_fact, steps + 1))
                                # Track which facts helped derive this one
                                substituted_conditions = [self.substitute_variable(cond, entity) 
                                                       for cond in conditions]
                                for cond in substituted_conditions:
                                    self.fact_graph[cond].append(new_fact)
        
        return self.derived_steps
    
    def get_derivation_path(self, fact: str) -> List[List[str]]:
        """Get all facts used in deriving a specific fact."""
        if fact not in self.derived_steps:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str]):
            if current in self.initial_facts:
                paths.append(path[:])
                return
            
            if current in visited:
                return
                
            visited.add(current)
            for prev_fact, next_facts in self.fact_graph.items():
                if current in next_facts:
                    dfs(prev_fact, path + [prev_fact])
            visited.remove(current)
        
        dfs(fact, [fact])
        return paths

# Example usage function
def analyze_theory(theory_data: Dict) -> Dict[str, List[Tuple[str, int, List[List[str]]]]]:
    """Analyze a theory and return new facts with their steps and derivation paths."""
    analyzer = TheoryAnalyzer(theory_data['facts'], theory_data['rules'])
    derived_steps = analyzer.analyze()
    
    # Organize results
    new_facts = {}
    for fact, steps in derived_steps.items():
        if fact not in theory_data['facts']:  # Only include newly derived facts
            paths = analyzer.get_derivation_path(fact)
            new_facts[fact] = (steps, paths)
            
    return new_facts