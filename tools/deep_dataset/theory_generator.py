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
            # 'Ian', 'Jack', 'Kevin', 'Lisa', 'Mary', 'Nancy', 'Oscar', 'Paul', 
            # 'Quinn', 'Robert', 'Sarah', 'Thomas', 'Uma', 'Victor', 'William', 
            # 'Xavier', 'Yolanda', 'Zach'
        ]
        self.type1_attributes = [
    'Able', 'Blue', 'Calm', 'Dark', 'Eager', 'Fast', 'Green', 'Happy', 
    'Intelligent', 'Joyful', 'Kind', 'Loud', 'Mighty', 'Nice',
#     'Observant', 
#     'Patient', 'Quick', 'Rough', 'Smart', 'Tall', 'Unique', 'Valiant', 
#     'Wise', 'Xenial', 'Young', 'Zealous'
]
        
        # Type 2 configuration
        self.type2_predicates = [
    'admires', 'believes', 'chases', 'doubts', 'envies', 'follows', 
    # 'guides', 'helps', 'ignores', 'joins', 'knows', 'leads', 'meets', 
    # 'notices', 'obeys', 'protects', 'questions', 'respects', 'supports', 
    # 'teaches', 'understands', 'visits', 'watches', 'xeroxes', 'yields', 'zeros'
]
        self.type2_entities = [
    'Antelope', 'Bear', 'Cat', 'Dog', 'Eagle', 'Fox', 'Goat', 'Horse', 
    'Ibex', 'Jaguar', 'Kangaroo', 
    # 'Lion', 'Mouse', 'Newt', 'Owl', 'Penguin', 
    # 'Quail', 'Rabbit', 'Snake', 'Tiger', 'Unicorn', 'Vulture', 'Wolf', 
    # 'Xerus', 'Yak', 'Zebra'
]
        self.type2_attributes = [
    'Agile', 'Brave', 'Clever', 'Dangerous', 'Elusive', 'Fierce', 'Graceful', 
    'Hungry', 'Instinctive', 'Jumpy', 
    # 'Keen', 'Large', 'Mighty', 'Nimble', 
    # 'Observant', 'Powerful', 'Quick', 'Robust', 'Swift', 'Tenacious', 
    # 'Untamed', 'Vigilant', 'Wild', 'Xerophilic', 'Young', 'Zealous'
]

    def negate(self, statement):
        if random.random() < 0.2:
            return '~'+statement
        else:
            return statement

    def generate_type1_theory(self) -> Dict:
        """Generate a Type 1 theory using only is() predicate in logical notation."""
        num_facts = random.randint(16, 32)
        # num_facts = 100
        num_rules = random.randint(8, 16)
        # num_rules = 100
        
        num_entities = random.randint(4, 8)
        num_attributes = random.randint(5, 10)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type1_entities, num_entities)
        attributes = random.sample(self.type1_attributes, num_attributes)
        
        # Generate facts in is(Entity,Attribute) format
        facts = []
        for _ in range(num_facts):
            entity = random.choice(entities)
            attribute = random.choice(attributes)
            fact = self.negate(f"is({entity},{attribute})")
            facts.append(fact)
            
        # Generate rules in logical notation
        rules = []
        for _ in range(num_rules):

            num_conditions = random.randint(2, 4)
            conditions = []

            used_arguments = set()

            # Select first argument of first condition, 20% chance for entity, 80% chance for variable
            if random.random() < 0.2:  
                argument = random.choice(entities)
            else:
                argument = "?X" 
                
            # Select attribute of first condition
            attr = random.choice(attributes)
            
            # Build first condition, 20% change for negation, 80% change for no negation
            condition = self.negate(f"is({argument},{attr})")
            conditions.append(condition)
                
            # Track used arguments
            used_arguments.add(argument)
        
            
            # Generate subsequent conditions
            for _ in range(num_conditions):
                # Select first argument from list of used arguments
                argument = random.choice(list(used_arguments))
                
                # Select attribute
                attr = random.choice(attributes)
            
                # Build condition, 20% change for negation, 80% change for no negation
                condition = self.negate(f"is({argument},{attr})")
                conditions.append(condition)
                used_arguments.add(argument)
            
            # Generate conclusion
            argument = random.choice(list(used_arguments))
            conclusion_attr = random.choice(attributes)
            conclusion = self.negate(f"is({argument},{conclusion_attr})")

            # Combine into rule with logical notation
            if len(conditions) == 1:
                rule = f"{conditions[0]} → {conclusion}."
            else:
                rule = " ∧ ".join(conditions) + f" → {conclusion}."
            
            rules.append(rule)
                
        return {'facts':facts, 'rules':rules}

    def generate_type2_theory(self, num_facts: int = None, num_rules: int = None) -> Dict:
        """Generate a Type 2 theory using is() and other predicates in logical notation."""
        
        raise NotImplementedError('only type one for now')
        
        num_facts = random.randint(15, 16)
        num_rules = random.randint(7, 8)
        
        num_entities = random.randint(2, 4)
        num_attributes = random.randint(1, 5)
        num_predicates = random.randint(1, 4)
            
        # Select random entities and attributes for this theory
        entities = random.sample(self.type2_entities, num_entities)
        attributes = random.sample(self.type2_attributes, num_attributes)
        predicates = random.sample(self.type2_predicates, num_predicates)
        
        # Generate facts
        facts = []
        for _ in range(num_facts):
            if random.random() < 0.3:  # 30% chance for is() predicate
                entity = random.choice(entities)
                attribute = random.choice(attributes)
                facts.append(f"is({entity},{attribute})")
            else:  # relation between entities
                predicate = random.choice(predicates)
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
                
            num_conditions = random.randint(1, )  # Max 5 conditions
            conditions = []
            # used_predicates = set()
            
            # Generate conditions
            for _ in range(num_conditions):
                if random.random() < 0.25:  # 50% chance for is() predicate
                    attr = random.choice(attributes)
                    conditions.append(f"is({variable},{attr})")
                else:
                    # available_preds = [p for p in self.type2_predicates if p not in used_predicates]
                    # if not available_preds:
                    #     continue
                    predicate = random.choice(predicates)
                    # used_predicates.add(predicate)
                    entity = random.choice(entities)
                    conditions.append(f"{predicate}({variable},{entity})")
            
            # Generate conclusion
            if random.random() < 0.25:
                conclusion_attr = random.choice(attributes)
                conclusion = f"is({variable},{conclusion_attr})"
            else:
                # available_preds = [p for p in self.type2_predicates if p not in used_predicates]
                # if not available_preds:
                #     predicate = random.choice(self.type2_predicates)
                # else:
                #     predicate = random.choice(available_preds)
                predicate = random.choice(predicates)
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
    # theory2 = generator.generate_type2_theory()
    
    d = {
        "Theory1": theory1,
        # "Theory2": theory2
    }
    
    # print(theory1)
    with open('out.json', 'w') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
