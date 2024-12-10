import random
from dataclasses import dataclass
from typing import List, Set, Dict
import json
# @dataclass
# class Theory:
#     facts: List[str]  # Still in is(Entity,Attribute) format
#     rules: List[str]  # Now in logical notation with ∧ and →

class LogicalTheoryGenerator:
    def __init__(self):
        # Type 1 configuration
        self.type1_entities = ['Alan', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Greg', 'Henry', 'Ian', 'John']
        self.type1_attributes = ['Blue', 'Rough', 'Young', 'Big', 'Round', 'Kind', 'Green', 'Nice', 'Smart', 
                               'Tall', 'Strong', 'Fast', 'Quiet', 'Wise']
        
        # Type 2 configuration
        self.type2_predicates = ['likes', 'chases', 'eats', 'follows']
        self.type2_entities = ['Cat', 'Dog', 'BaldEagle', 'Mouse', 'Rabbit', 'Bird', 'Fox', 'Wolf', 'Bear', 'Lion']
        self.type2_attributes = ['Big', 'Furry', 'Fast', 'Strong', 'Wild', 'Fierce', 'Brave', 'Quiet', 'Smart', 'Wise']

    def generate_type1_theory(self, num_facts: int = None, num_rules: int = None) -> Dict:
        """Generate a Type 1 theory using only is() predicate in logical notation."""
        if num_facts is None:
            num_facts = random.randint(1, 16)
        if num_rules is None:
            num_rules = random.randint(1, 9)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type1_entities, min(4, len(self.type1_entities)))
        attributes = random.sample(self.type1_attributes, min(7, len(self.type1_attributes)))
        
        # Generate facts in is(Entity,Attribute) format
        facts = []
        for _ in range(num_facts):
            entity = random.choice(entities)
            attribute = random.choice(attributes)
            facts.append(f"is({entity},{attribute})")
            
        # Generate rules in logical notation
        rules = []
        for _ in range(num_rules):
            if random.random() < 0.2:  # 20% chance for grounded rule
                variable = random.choice(entities)
            else:
                variable = "?X"
                
            num_conditions = random.randint(1, 2)
            conditions = []
            used_attributes = set()
            
            # Generate conditions
            for _ in range(num_conditions):
                available_attrs = [a for a in attributes if a not in used_attributes]
                if not available_attrs:
                    break
                attr = random.choice(available_attrs)
                used_attributes.add(attr)
                conditions.append(f"is({variable},{attr})")
            
            # Generate conclusion with unused attribute
            available_attrs = [a for a in attributes if a not in used_attributes]
            if not available_attrs:
                continue
            conclusion_attr = random.choice(available_attrs)
            conclusion = f"is({variable},{conclusion_attr})"
            
            # Combine into rule with logical notation
            if len(conditions) == 1:
                rule = f"{conditions[0]} → {conclusion}."
            else:
                rule = f"{conditions[0]} ∧ {conditions[1]} → {conclusion}."
            
            rules.append(rule)
                
        return {'facts':facts, 'rules':rules}

    def generate_type2_theory(self, num_facts: int = None, num_rules: int = None) -> Dict:
        """Generate a Type 2 theory using is() and other predicates in logical notation."""
        if num_facts is None:
            num_facts = random.randint(1, 16)
        if num_rules is None:
            num_rules = random.randint(1, 9)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type2_entities, min(4, len(self.type2_entities)))
        attributes = random.sample(self.type2_attributes, min(5, len(self.type2_attributes)))
        
        # Generate facts
        facts = []
        for _ in range(num_facts):
            if random.random() < 0.5:  # 50% chance for is() predicate
                entity = random.choice(entities)
                attribute = random.choice(attributes)
                facts.append(f"is({entity},{attribute})")
            else:  # relation between entities
                predicate = random.choice(self.type2_predicates)
                entity1 = random.choice(entities)
                entity2 = random.choice([e for e in entities if e != entity1])
                facts.append(f"{predicate}({entity1},{entity2})")
                
        # Generate rules
        rules = []
        for _ in range(num_rules):
            if random.random() < 0.2:  # 20% chance for grounded rule
                variable = random.choice(entities)
            else:
                variable = "?X"
                
            num_conditions = random.randint(1, 2)
            conditions = []
            used_predicates = set()
            
            # Generate conditions
            for _ in range(num_conditions):
                if random.random() < 0.5:  # 50% chance for is() predicate
                    attr = random.choice(attributes)
                    conditions.append(f"is({variable},{attr})")
                else:
                    available_preds = [p for p in self.type2_predicates if p not in used_predicates]
                    if not available_preds:
                        continue
                    predicate = random.choice(available_preds)
                    used_predicates.add(predicate)
                    entity = random.choice(entities)
                    conditions.append(f"{predicate}({variable},{entity})")
            
            # Generate conclusion
            if random.random() < 0.5:
                conclusion_attr = random.choice(attributes)
                conclusion = f"is({variable},{conclusion_attr})"
            else:
                available_preds = [p for p in self.type2_predicates if p not in used_predicates]
                if not available_preds:
                    predicate = random.choice(self.type2_predicates)
                else:
                    predicate = random.choice(available_preds)
                entity = random.choice(entities)
                conclusion = f"{predicate}({variable},{entity})"
            
            # Combine into rule with logical notation
            if len(conditions) == 1:
                rule = f"{conditions[0]} → {conclusion}."
            else:
                rule = f"{conditions[0]} ∧ {conditions[1]} → {conclusion}."
            
            rules.append(rule)
                
        return {'facts':facts, 'rules':rules}

# Example usage
if __name__ == "__main__":
    generator = LogicalTheoryGenerator()
    
    # print(f'##### Round {i} #######')
    # Generate and print a Type 1 theory
    print("=== Type 1 Theory ===")
    theory1 = generator.generate_type1_theory()
    
    # print(theory1)
    with open('out.json', 'w') as f:
        json.dump(theory1, f, ensure_ascii=False, indent=2)
    # print("Facts:")
    # for fact in theory1.facts:
    #     print(fact)
    # print("\nRules:")
    # for rule in theory1.rules:
    #     print(rule)
        
    # print("\n=== Type 2 Theory ===")
    # theory2 = generator.generate_type2_theory()
    # print("Facts:")
    # for fact in theory2.facts:
    #     print(fact)
    # print("\nRules:")
    # for rule in theory2.rules:
    #     print(rule)