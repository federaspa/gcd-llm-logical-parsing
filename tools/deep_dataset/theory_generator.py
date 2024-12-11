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
        self.type1_entities = [
            'Alan', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'George', 'Henry', 
            'Ian', 'Jack', 'Kevin', 'Lisa', 'Mary', 'Nancy', 'Oscar', 'Paul', 
            'Quinn', 'Robert', 'Sarah', 'Thomas', 'Uma', 'Victor', 'William', 
            'Xavier', 'Yolanda', 'Zach'
        ]
        self.type1_attributes = [
    'Able', 'Blue', 'Calm', 'Dark', 'Eager', 'Fast', 'Green', 'Happy', 
    'Intelligent', 'Joyful', 'Kind', 'Loud', 'Mighty', 'Nice', 'Observant', 
    'Patient', 'Quick', 'Rough', 'Smart', 'Tall', 'Unique', 'Valiant', 
    'Wise', 'Xenial', 'Young', 'Zealous'
]
        
        # Type 2 configuration
        self.type2_predicates = [
    'admires', 'believes', 'chases', 'doubts', 'envies', 'follows', 
    'guides', 'helps', 'ignores', 'joins', 'knows', 'leads', 'meets', 
    'notices', 'obeys', 'protects', 'questions', 'respects', 'supports', 
    'teaches', 'understands', 'visits', 'watches', 'xeroxes', 'yields', 'zeros'
]
        self.type2_entities = [
    'Antelope', 'Bear', 'Cat', 'Dog', 'Eagle', 'Fox', 'Goat', 'Horse', 
    'Ibex', 'Jaguar', 'Kangaroo', 'Lion', 'Mouse', 'Newt', 'Owl', 'Penguin', 
    'Quail', 'Rabbit', 'Snake', 'Tiger', 'Unicorn', 'Vulture', 'Wolf', 
    'Xerus', 'Yak', 'Zebra'
]
        self.type2_attributes = [
    'Agile', 'Brave', 'Clever', 'Dangerous', 'Elusive', 'Fierce', 'Graceful', 
    'Hungry', 'Instinctive', 'Jumpy', 'Keen', 'Large', 'Mighty', 'Nimble', 
    'Observant', 'Powerful', 'Quick', 'Robust', 'Swift', 'Tenacious', 
    'Untamed', 'Vigilant', 'Wild', 'Xerophilic', 'Young', 'Zealous'
]

    def generate_type1_theory(self, num_facts: int = None, num_rules: int = None) -> Dict:
        """Generate a Type 1 theory using only is() predicate in logical notation."""
        if num_facts is None:
            num_facts = 40
            # num_facts = random.randint(1, 16)
        if num_rules is None:
            num_rules = 20
            # num_rules = random.randint(1, 9)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type1_entities, min(8, len(self.type1_entities)))
        attributes = random.sample(self.type1_attributes, min(14, len(self.type1_attributes)))
        
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
                
            num_conditions = random.randint(1, 5)
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
                rule = " ∧ ".join(conditions) + f" → {conclusion}."
            
            rules.append(rule)
                
        return {'facts':facts, 'rules':rules}

    def generate_type2_theory(self, num_facts: int = None, num_rules: int = None) -> Dict:
        """Generate a Type 2 theory using is() and other predicates in logical notation."""
        if num_facts is None:
            num_facts = 40
            # num_facts = random.randint(1, 16)
        if num_rules is None:
            num_rules = 20
            # num_rules = random.randint(20, 21)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type2_entities, min(8, len(self.type2_entities)))
        attributes = random.sample(self.type2_attributes, min(10, len(self.type2_attributes)))
        
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
                
            num_conditions = random.randint(1, 5)  # Max 5 conditions
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
                rule = " ∧ ".join(conditions) + f" → {conclusion}."
            
            rules.append(rule)
                
        return {'facts':facts, 'rules':rules}

# Example usage
if __name__ == "__main__":
    generator = LogicalTheoryGenerator()

    theory1 = generator.generate_type1_theory()
    theory2 = generator.generate_type2_theory()
    
    d = {
        "Theory1": theory1,
        "Theory2": theory2
    }
    
    # print(theory1)
    with open('out.json', 'w') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
