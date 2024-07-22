import re
import json
import random

import pandas as pd


def quantifier_misplacement(formula):
    # Find all quantifiers and logical operators in the formula
    quantifiers = re.findall(r'[∀∃]\w+', formula)
    operators = re.findall(r'[∧∨→↔]', formula)
    
    if not quantifiers or not operators:
        return formula  # Can't perform the operation if there are no quantifiers or operators
    
    # Choose a random quantifier and operator
    quantifier = random.choice(quantifiers)
    operator = random.choice(operators)
    
    # Remove the chosen quantifier from its original position
    formula = formula.replace(quantifier, '', 1)
    
    # Find the position of the chosen operator
    operator_position = formula.index(operator)
    
    # Insert the quantifier right after the operator
    new_formula = formula[:operator_position + 1] + ' ' + quantifier + ' ' + formula[operator_position + 1:]
    
    return new_formula

def parentheses_manipulation(formula):
    if '(' not in formula or ')' not in formula:
        return formula
    
    open_positions = [i for i, char in enumerate(formula) if char == '(']
    close_positions = [i for i, char in enumerate(formula) if char == ')']
    
    if random.choice([True, False]):
        # Remove a random pair of parentheses
        remove_index = random.randint(0, min(len(open_positions), len(close_positions)) - 1)
        formula = formula[:open_positions[remove_index]] + formula[open_positions[remove_index]+1:]
        formula = formula[:close_positions[remove_index]-1] + formula[close_positions[remove_index]:]
    else:
        # Add a pair of parentheses
        start = random.randint(0, len(formula) - 2)
        end = random.randint(start + 1, len(formula) - 1)
        formula = formula[:start] + '(' + formula[start:end] + ')' + formula[end:]
    
    return formula

def variable_scope_alteration(formula):
    quantifiers = re.findall(r'[∀∃]\w+', formula)
    if len(quantifiers) < 2:
        return formula
    
    q1, q2 = random.sample(quantifiers, 2)
    formula = formula.replace(q1, '@TEMP@')
    formula = formula.replace(q2, q1)
    formula = formula.replace('@TEMP@', q2)
    
    return formula

def quantifier_removal(formula):
    quantifiers = re.findall(r'[∀∃]\w+', formula)
    if not quantifiers:
        return formula
    
    quantifier = random.choice(quantifiers)
    return formula.replace(quantifier, '', 1)

def no_error(formula):
    return formula

def introduce_error(formula):
    error_functions = [
        quantifier_misplacement,
        parentheses_manipulation,
        variable_scope_alteration,
        quantifier_removal,
        no_error
    ]
    
    chosen_function = random.choice(error_functions)
    return str(chosen_function.__name__), chosen_function(formula)

def create_training_example(item):
    
    nl_statement = random.choice(item['context'])
    correct = item['context_fol'][item['context'].index(nl_statement)]
    predicates = "\n".join(item['logic_predicates'])
    func, sketch = introduce_error(correct)
    
    return {
        "NLSTATEMENT": nl_statement,
        "ERRORTYPE": func,
        "SKETCH": sketch,
        "CORRECT": correct,
        "PREDICATES": predicates
    }



def create_prompt(example):
    system_prompt = """Given a Natural Language Statement, a Sketch First Order Logic Statement and a set of Predicates. 
The task is to rewrite the Sketch using only the provided Predicates, and fix errors if any, following the provided examples.

For fixing the formulas:
1. You SHOULD ONLY USE the predicates provided.
2. You SHOULD USE the following logical operators: ⊕ (either or), ∨ (disjunction),
∧ (conjunction), → (implication), ∀ (universal), ∃ (existential), ¬ (negation),
↔ (equivalence)
3. You SHOULD NEVER USE the following symbols for FOL: "%", "≠"
4. The literals in FOL SHOULD ALWAYS have predicate and entities, e.g., "Rounded(x, y)" or "City(guilin)";
expressions such as "y = a ∨ y = b" or "a ∧ b ∧ c" are NOT ALLOWED
5. The FOL rule SHOULD ACCURATELY reflect the meaning of the corresponding NL statement
6. The FOL rule SHOULD BE AS CLOSE AS POSSIBLE to the provided sketch
7. You SHOULD NEVER have an entity with the same name as a predicate, e.g., Chair(x) and Tall(chair) is NOT ALLOWED
------
"""

    user_prompt = f"""Natural Language Statement: {example['NLSTATEMENT']}

Sketch First Order Logic Statement: {example['SKETCH']}

Predicates:
\"\"\"
{example['PREDICATES']}
\"\"\"
######
Valid FOL Statement: {example['CORRECT']}
"""

    return f"{system_prompt}\n{user_prompt}"

def main():
    with open('data/FOLIO/train_original.json', 'r') as f:
        data = json.load(f)
        
    training_data = []
    
    for item in data:
        try:
            training_data.append(create_training_example(item))
        except:
            continue
    
    with open('train_finetune.json', 'w') as f:
        json.dump(training_data, f, indent=4, ensure_ascii=False)
        
        
    training_prompts = '\n<s>'.join([create_prompt(example) for example in training_data])
    
    # # Create a DataFrame with prompts and empty responses
    # df = pd.DataFrame({
    #     'prompt': training_prompts,
    #     'response': [p['CORRECT'] for p in training_data]
    # })

    # # Save the DataFrame as a CSV file
    # df.to_csv('llama2_finetuning_data.csv', index=False)
    
    with open('train_finetune_prompts.txt', 'w') as f:
        f.write(training_prompts)

        
        
        
        
        
        
if __name__ == '__main__':
    main()