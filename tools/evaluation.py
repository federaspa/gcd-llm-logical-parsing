import re
import json
import os
import argparse
import numpy as np

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
    def compute_metrics(samples, with_string=False):
        def compute_ratio(numerator, denominator, with_string):
            ratio = numerator/denominator if denominator else 0
            return (ratio, f'{numerator}/{denominator}') if with_string else ratio

        parsable_samples = [s for s in samples if s['status'] != 'parsing error']
        executable_samples = [s for s in samples if s['status'] == 'success']
        
        gold_answers, predictions = LogicEvaluator.parse_answers(samples)
        correct_predictions = sum(g == p for g, p in zip(gold_answers, predictions))
        
        metrics = [
            compute_ratio(correct_predictions, len(samples), with_string),
            compute_ratio(sum(g == p for g, p in zip(*LogicEvaluator.parse_answers(executable_samples))), 
                         len(executable_samples), with_string),
            compute_ratio(len(parsable_samples), len(samples), with_string),
            compute_ratio(len(executable_samples), len(samples), with_string)
        ]
        
        return metrics

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
                
                unconstrained = [s['logic_problem'] for s in success_samples if 'logic_problem' in s]
                constrained = [s['logic_problem_gcd'] for s in success_samples if 'logic_problem_gcd' in s]
            
            elif category == 'EXCLUSIVE_SUCCESS':
                unc_samples = filter_samples(
                    lambda s: get_status(s, 'logic_problem') == 'success' != get_status(s, 'logic_problem_gcd')
                )
                con_samples = filter_samples(
                    lambda s: get_status(s, 'logic_problem_gcd') == 'success' != get_status(s, 'logic_problem')
                )
                unconstrained = [s['logic_problem'] for s in unc_samples if 'logic_problem' in s]
                constrained = [s['logic_problem_gcd'] for s in con_samples if 'logic_problem_gcd' in s]
            
            elif category == 'NEITHER_SUCCESS':
                unsuccess_samples = filter_samples(
                    lambda s: (get_status(s, 'logic_problem') != 'success') and (get_status(s, 'logic_problem_gcd') != 'success')
                )
            
                unconstrained = [s['logic_problem'] for s in unsuccess_samples if 'logic_problem' in s]
                constrained = [s['logic_problem_gcd'] for s in unsuccess_samples if 'logic_problem_gcd' in s]
            
        return LogicEvaluator.compute_metrics(unconstrained, with_string), LogicEvaluator.compute_metrics(constrained, with_string)

    @staticmethod
    def format_latex_table(results, table_name=""):
        """
        Format results in a LaTeX table similar to the provided example.
        
        Args:
            results (dict): Results dictionary containing metrics
            table_name (str): Optional name for the table caption
        
        Returns:
            str: LaTeX formatted table
        """
        latex_lines = [
            "\\begin{table}[t]",
            "\\begin{tabular}{l|ccc|ccc}",
            "\\hline",
            " & \\multicolumn{3}{c|}{Unconstrained} & \\multicolumn{3}{c}{Constrained} \\\\ \\hline",
            "Model & Total & Covered & Full & Total & Covered & Full \\\\",
            " & Acc. & Acc. & Cov. & Acc. & Acc. & Cov. \\\\",
            "\\hline"
        ]
        
        # Process each model's results
        for model_name, model_results in results.items():
            if 'ALL' not in model_results:
                continue
                
            unc_metrics = model_results['ALL']['UNCONSTRAINED']
            con_metrics = model_results['ALL']['CONSTRAINED']
            
            # Format each metric as a percentage with 2 decimal places
            metrics_line = [
                model_name,
                f"{unc_metrics[0][0]:.2f}",  # Total Acc Unconstrained
                f"{unc_metrics[1][0]:.2f}",  # Covered Acc Unconstrained
                f"{unc_metrics[3][0]:.2f}",  # Full Cov Unconstrained
                f"{con_metrics[0][0]:.2f}",  # Total Acc Constrained
                f"{con_metrics[1][0]:.2f}",  # Covered Acc Constrained
                f"{con_metrics[3][0]:.2f}"   # Full Cov Constrained
            ]
            latex_lines.append(" & ".join(metrics_line) + " \\\\" + " \\hline")
        
        # Add table footer
        latex_lines.extend([
            "\\end{tabular}"
        ])
        
        # Add caption if provided
        if table_name:
            latex_lines.append(f"\\caption{{{table_name}}}")
            
        latex_lines.append("\\end{table}")
        
        return "\n".join(latex_lines)

def print_results(results, categories, with_string=False, latex_output=False):
    metric_names = ['Total accuracy', 'Accuracy', 'Parsing Coverage', 'Coverage']
    
    if latex_output:
        print(LogicEvaluator.format_latex_table(results))
        return
        
    for category in categories:
        for subcategory in ['UNCONSTRAINED', 'CONSTRAINED']:
            print(f'{"-"*20}\n{subcategory}\n{"-"*20}\n')
            
            metrics = results[category][subcategory]
            for name, value in zip(metric_names, metrics):
                if name in ['Parsing Coverage']:
                    continue
                
                if with_string:
                    print(f'{name}: {value[0]:.2f} ({value[1]})\n')
                else:
                    print(f'{name}: {value:.2f}\n')

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

            if not args.latex:
                print(f'{"#"*20}\n{sketcher_name}\n{"#"*20}\n')
            
            with open(result_file, 'r') as f:
                samples = json.load(f)

            results = {}
            categories = ['ALL']

            for category in categories:
                results[category] = {}
                unc_metrics, con_metrics = LogicEvaluator.evaluate_sample_groups(samples, category, True)
                results[category]['UNCONSTRAINED'] = unc_metrics
                results[category]['CONSTRAINED'] = con_metrics
            
            all_results[sketcher_name] = results
            
            if not args.latex:
                print_results(results, categories, True)
            
        except Exception as e:
            continue
    
    if args.latex:
        print_results(all_results, categories, True, latex_output=True)

if __name__ == "__main__":
    main()