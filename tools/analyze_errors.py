import re
import json
import os
import argparse
import numpy as np

class LogicEvaluator:

    @staticmethod
    def evaluate_sample_groups(samples, category, with_string=False):
        def filter_samples(condition):
            return [s for s in samples if condition(s)]
            
        if category == 'ALL':
            unconstrained = [s['logic_problem'] for s in samples if 'logic_problem' in s]
            constrained = [s['logic_problem_gcd'] for s in samples if 'logic_problem_gcd' in s]
        else:
            def get_status(s, key):
                return s.get(key, {}).get('status', '')
                
            if category == 'BOTH_SUCCESS':
                success_samples = filter_samples(
                    lambda s: get_status(s, 'logic_problem') == get_status(s, 'logic_problem_gcd') == 'success'
                )
                
                unconstrained = [s['logic_problem'] for s in success_samples]
                constrained = [s['logic_problem_gcd'] for s in success_samples]
            
            elif category == 'EXCLUSIVE_SUCCESS':
                unc_samples = filter_samples(
                    lambda s: get_status(s, 'logic_problem') != get_status(s, 'logic_problem_gcd') == 'success'
                )
                con_samples = filter_samples(
                    lambda s: get_status(s, 'logic_problem_gcd') != get_status(s, 'logic_problem') == 'success'
                )
                unconstrained = [s['logic_problem'] for s in unc_samples]
                constrained = [s['logic_problem_gcd'] for s in con_samples]
            
            elif category == 'NEITHER_SUCCESS':
                unsuccess_samples = filter_samples(
                    lambda s: (get_status(s, 'logic_problem') != 'success') and (get_status(s, 'logic_problem_gcd') != 'success')
                )
            
                unconstrained = [s['logic_problem'] for s in unsuccess_samples]
                constrained = [s['logic_problem_gcd'] for s in unsuccess_samples]
                
        return unconstrained, constrained

def examine_samples(samples, category):
    unconstrained, constrained = LogicEvaluator.evaluate_sample_groups(samples, category)
    print(f"\nCategory: {category}")
    print(f"Number of samples - Unconstrained: {len(unconstrained)}, Constrained: {len(constrained)}")
    
    while True:
        print("\nOptions:")
        print("1. View unconstrained samples")
        print("2. View constrained samples")
        print("3. Return to category selection")
        choice = input("Enter your choice (1-3): ")
        
        if choice == "3":
            break
            
        if choice not in ["1", "2"]:
            print("Invalid choice. Please try again.")
            continue
            
        sample_list = unconstrained if choice == "1" else constrained
        problem_type = "logic_problem" if choice == "1" else "logic_problem_gcd"
        
        for i, sample in enumerate(sample_list):
            print(f"\nSample {i+1}/{len(sample_list)}")
            print("Error:", sample.get("error", "No error information available"))
            print("\nRules:")
            print('\n'.join(sample.get("fol_rules", ["No program available"])))
            print("\nProgram:")
            print(sample)
            
            action = input("\nPress Enter for next sample, 'b' to go back, or 'q' to quit: ").lower()
            if action == 'b':
                break
            elif action == 'q':
                return False
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--result-path', type=str, default='./outputs/logic_inference')
    args = parser.parse_args()
    
    prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
    filename = f'{prefix}{args.dataset_name}_{args.split}_{args.sketcher_name}.json'
    result_file = os.path.join(args.result_path, filename)
    
    print('Evaluating file\n', result_file)
    with open(result_file, 'r') as f:
        samples = json.load(f)
    
    categories = ['ALL', 'BOTH_SUCCESS', 'EXCLUSIVE_SUCCESS', 'NEITHER_SUCCESS']
    
    while True:
        print("\nAvailable categories:")
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")
        print("5. Exit")
        
        try:
            choice = int(input("\nSelect category (1-5): "))
            if choice == 5:
                break
            if 1 <= choice <= 4:
                if not examine_samples(samples, categories[choice-1]):
                    break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()