import re
import json
import os
import argparse
import numpy as np
import pandas as pd

class LogicEvaluator:
    CHOICES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N/A']
    CHOICE_PATTERNS = [f"{c}{s}" for c in CHOICES for s in ['', ')', '.']]
    ANSWER_INDICATORS = [
        'the correct option is', 'the correct answer is',
        'The correct answer is', 'The correct option is',
        'Thus, the answer is'
    ]

    @staticmethod
    def get_choice(answer_str):
        for pattern in LogicEvaluator.CHOICE_PATTERNS:
            if answer_str.startswith(pattern):
                return pattern.replace(')', '').replace('.', '')
        
        for indicator in LogicEvaluator.ANSWER_INDICATORS:
            if indicator in answer_str:
                answer_str = answer_str.split(indicator)[1].strip()
                for pattern in LogicEvaluator.CHOICE_PATTERNS:
                    if answer_str.startswith(pattern):
                        return pattern.replace(')', '').replace('.', '')
        return None

    @staticmethod
    def parse_answers(samples):
        gold_answers = []
        predictions = []
        
        for sample in samples:
            gold_answer = sample['answer']
            prediction = LogicEvaluator.get_choice(sample['predicted_answer'].strip())
            gold_answers.append(gold_answer)
            predictions.append(prediction)
            
        return gold_answers, predictions

    @staticmethod
    def compute_metrics(samples, n, with_string=False):
        def compute_ratio(numerator, denominator, with_string):
            ratio = numerator/denominator if denominator else 0
            return (ratio, f'{numerator}/{denominator}') if with_string else ratio

        parsable_samples = [s for s in samples if s['status'] != 'parsing error']
        executable_samples = [s for s in samples if s['status'] == 'success']
        
        gold_answers, predictions = LogicEvaluator.parse_answers(samples)
        correct_predictions = sum(g == p for g, p in zip(gold_answers, predictions))
        
        metrics = [
            compute_ratio(correct_predictions, n, with_string),
            compute_ratio(sum(g == p for g, p in zip(*LogicEvaluator.parse_answers(executable_samples))), 
                         len(executable_samples), with_string),
            compute_ratio(len(parsable_samples), n, with_string),
            compute_ratio(len(executable_samples), n, with_string)
        ]
        
        return metrics
    
    @staticmethod
    def compute_baseline_metrics(samples, with_string=False):
        def compute_ratio(numerator, denominator, with_string):
            ratio = numerator/denominator if denominator else 0
            return (ratio, f'{numerator}/{denominator}') if with_string else ratio
        
        gold_answers = [s['nl_problem']['answer'] for s in samples]

        random_answers = np.random.choice(['A', 'B', 'C', 'N/A'], len(samples))
        fixed_answers = ['A']*len(samples)
        correct_random = sum(g == p for g, p in zip(gold_answers, random_answers))
        correct_fixed = sum(g == p for g, p in zip(gold_answers, fixed_answers))
        
        return compute_ratio(correct_random, len(samples), with_string), compute_ratio(correct_fixed, len(samples), with_string)

    @staticmethod
    def evaluate_sample_groups(samples, with_string=False):
        def filter_samples(condition):
            return [s for s in samples if condition(s)]
        
        n = len(samples)
            
        unconstrained = [s['logic_problem'] for s in samples if 'logic_problem' in s]
        constrained = [s['logic_problem_gcd'] for s in samples if 'logic_problem_gcd' in s]
        json = [s['logic_problem_json'] for s in samples if 'logic_problem_json' in s]
            
        return LogicEvaluator.compute_metrics(unconstrained, n, with_string), LogicEvaluator.compute_metrics(constrained, n, with_string), LogicEvaluator.compute_metrics(json, n, with_string)

def print_results(results):
    """
    Format results in a pandas DataFrame with hierarchical columns.
    
    Args:
        results (dict): Results dictionary containing metrics for different models
    """
    # Define metric names
    metric_names = ['Overall Accuracy', 'Executable Accuracy', 'Parse Rate', 'Execution Rate']
    
    # Create MultiIndex for columns
    categories = ['Unconstrained', 'JSON', 'Constrained']
    column_tuples = [(cat, metric) for cat in categories for metric in ['Accuracy', 'Coverage']]
    columns = pd.MultiIndex.from_tuples(column_tuples)
    
    # Initialize data dictionary
    data = {}
    
    # Process each model's results
    for model_name, model_results in sorted(results.items()):
        row_data = []
        
        # Get metrics for each category
        for category in ['UNCONSTRAINED', 'JSON', 'CONSTRAINED']:
            metrics = model_results[category]
            # Add accuracy (first metric) and coverage (last metric)
            row_data.extend([
                metrics[0][0] if isinstance(metrics[0], tuple) else metrics[0],  # Accuracy
                metrics[3][0] if isinstance(metrics[3], tuple) else metrics[3]   # Coverage
            ])
        
        data[model_name] = row_data
    
    # Create DataFrame
    df = pd.DataFrame(data).T
    df.columns = columns
    
    # Format as percentages
    df = df.round(4) * 100
    
    # Add baseline results if available
    if 'RANDOM' in next(iter(results.values())):
        baseline_data = {}
        for model_name, model_results in results.items():
            if model_name == next(iter(results.keys())):  # Only need to do this once
                random_acc = model_results['RANDOM'][0] if isinstance(model_results['RANDOM'], tuple) else model_results['RANDOM']
                fixed_acc = model_results['FIXED'][0] if isinstance(model_results['FIXED'], tuple) else model_results['FIXED']
                baseline_data['Random Baseline'] = [random_acc * 100] * len(df.columns)
                baseline_data['Fixed Baseline'] = [fixed_acc * 100] * len(df.columns)
        
        baseline_df = pd.DataFrame(baseline_data).T
        baseline_df.columns = df.columns
        df = pd.concat([df, baseline_df])
    
    # Print with formatting
    pd.set_option('display.float_format', '{:.2f}%'.format)
    print("\nResults:")
    print("="*80)
    print(df)
    print("="*80)
    
    return df

def print_latex_table(results, table_name=""):
    """
    Format results in a LaTeX table similar to the provided example.
    Highlights the highest value in each metric category (Accuracy and Coverage) with bold text.
    
    Args:
        results (dict): Results dictionary containing metrics
        table_name (str): Optional name for the table caption
    
    Returns:
        str: LaTeX formatted table
    """
    
    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\hline",
        "Model & \\multicolumn{3}{c|}{Accuracy} & \\multicolumn{3}{c}{Coverage}\\\\ \\hline ",
        " & Unc & Json & Con & Unc & Json & Con \\\\",
        "\\hline"
    ]
    
    # Sort results by key in alphabetical order
    results = dict(sorted(results.items()))
    
    # Process each model's results
    for model_name, model_results in results.items():
        unc_metrics = model_results['UNCONSTRAINED']
        con_metrics = model_results['CONSTRAINED']
        json_metrics = model_results['JSON']
        
        # Get accuracy values for comparison
        acc_values = [
            unc_metrics[0][0],
            json_metrics[0][0],
            con_metrics[0][0]
        ]
        max_acc = max(acc_values)
        
        # Get coverage values for comparison
        cov_values = [
            unc_metrics[3][0],
            json_metrics[3][0],
            con_metrics[3][0]
        ]
        max_cov = max(cov_values)
        
        # Format metrics with bold for highest values
        formatted_metrics = [
            model_name,
            f"\\textbf{{{unc_metrics[0][0]:.2f}}}" if unc_metrics[0][0] == max_acc else f"{unc_metrics[0][0]:.2f}",
            f"\\textbf{{{json_metrics[0][0]:.2f}}}" if json_metrics[0][0] == max_acc else f"{json_metrics[0][0]:.2f}",
            f"\\textbf{{{con_metrics[0][0]:.2f}}}" if con_metrics[0][0] == max_acc else f"{con_metrics[0][0]:.2f}",
            f"\\textbf{{{unc_metrics[3][0]:.2f}}}" if unc_metrics[3][0] == max_cov else f"{unc_metrics[3][0]:.2f}",
            f"\\textbf{{{json_metrics[3][0]:.2f}}}" if json_metrics[3][0] == max_cov else f"{json_metrics[3][0]:.2f}",
            f"\\textbf{{{con_metrics[3][0]:.2f}}}" if con_metrics[3][0] == max_cov else f"{con_metrics[3][0]:.2f}"
        ]
        
        latex_lines.append(" & ".join(formatted_metrics) + " \\\\" + " \\hline")
        
    # Add table footer
    latex_lines.extend([
        "\\end{tabular}"
    ])
    
    # Add caption if provided
    if table_name:
        latex_lines.append(f"\\caption{{{table_name}}}")
        
    latex_lines.append("\\end{table}")
    
    print("\n".join(latex_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--sketcher-name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--result-path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--latex', action='store_true', help='Output results in LaTeX table format')
    args = parser.parse_args()
    
    if args.sketcher_name:
        sketcher_names = [args.sketcher_name]
    else:
        config_path = './configs/models'
        sketcher_names = [os.path.splitext(f)[0] for f in os.listdir(config_path) if os.path.isfile(os.path.join(config_path, f))]
    
    all_results = {}
    for sketcher_name in sketcher_names:
        args.sketcher_name = sketcher_name
        
        try:
            prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
            filename = f'{prefix}{args.dataset_name}_{args.split}_{args.sketcher_name}.json'
            result_file = os.path.join(args.result_path, filename)
            
            with open(result_file, 'r') as f:
                samples = json.load(f)

            results = {}
            
            unc_metrics, con_metrics, json_metrics = LogicEvaluator.evaluate_sample_groups(samples, True)
            results['UNCONSTRAINED'] = unc_metrics
            results['CONSTRAINED'] = con_metrics
            results['JSON'] = json_metrics
            results['RANDOM'], results['FIXED'] = LogicEvaluator.compute_baseline_metrics(samples, True)

            all_results[sketcher_name] = results
            
        except Exception as e:
            print(e)
            continue
            
        
    if args.latex:
        print_latex_table(all_results)
    else:
        print_results(all_results)

if __name__ == "__main__":
    main()