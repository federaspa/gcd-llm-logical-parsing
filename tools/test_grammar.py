import json
from utils import OSModel
import argparse
import random
import re

class FOLGenerator:
    def __init__(self, args):
        self.args = args
        self.model_api = OSModel(args.model_path)
        self.grammar_path = f'./LLMs/grammars/FOL.gbnf'
        self.programs_path = args.programs_path
        self.load_grammar()
        self.dataset = self.load_raw_dataset()
        
    def load_grammar(self):
        with open(self.grammar_path, 'r') as f:
            self.grammar = f.read()

    def generate_fol_formulas(self):
        
        sample = random.choice(self.dataset)
        
        user_prompt, grammar = self.create_prompt(sample)
        
        response = self.model_api.invoke(
            user=user_prompt,
            task_description="You are a helpfull assistant.",
            raw_grammar=grammar,
            temperature=0.5,
            top_p=1,
            top_k=0,
            min_p=0.1,
            tfs_z=1,
            repeat_penalty=1.5,
            max_tokens = 150
        )
        return user_prompt, grammar, response['choices'][0]['message']['content']

    def load_raw_dataset(self):
        with open(self.programs_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def create_prompt(self, sample: dict) -> tuple[str, str | None]:
        
        premises = '\n'.join(sample['context_fol'])

        full_prompt = 'Generate a FOL formula like these ones:\n' + premises + '\n\nbut make it deeply nested'
        grammar = self.get_grammar(sample)

        return full_prompt, grammar

    def get_grammar(self, logic_problem: dict) -> str:
        
        predicates = []
        for formula in logic_problem['context_fol']:
            predicates.extend((re.findall(r'[A-Z][a-z]+\(', formula)))
            
        predicates = list(set(predicates))
        predicates = ' | '.join(f'"{pred.split("(")[0]}"' for pred in predicates)
        
        print(f"Predicates: {predicates}")
        
        return self.grammar.replace('[[PREDICATES]]', predicates)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FOL formulas using LLM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--programs_path', type=str, required=True, help='Path to the programs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    fol_generator = FOLGenerator(args)

    while True:
        input("Press enter to generate a new FOL formula")
        user_prompt, grammar, response = fol_generator.generate_fol_formulas()
        print(f"Response: {response}\n")