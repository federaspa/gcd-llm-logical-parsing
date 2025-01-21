import argparse
import json
import os
from core.logic_evaluator import LogicEvaluator
from utils.formatters import ResultFormatter

def save_results(results, args):
    """
    Save results to files in the output directory.
    
    Args:
        results (dict): Dictionary containing evaluation results
        args (argparse.Namespace): Command line arguments
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.save_path, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename
    prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
    base_filename = f'{prefix}{args.dataset_name}_{args.split}'
    
    # Save DataFrame results
    df = ResultFormatter.create_dataframe(results)
    csv_path = os.path.join(output_dir, f'{base_filename}_results.csv')
    df.to_csv(csv_path)
    
    # Save LaTeX results
    latex_content = ResultFormatter.format_latex(results)
    latex_path = os.path.join(output_dir, f'{base_filename}_results.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    
    # Save raw results as JSON for potential future use
    json_path = os.path.join(output_dir, f'{base_filename}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    
    return csv_path, latex_path, json_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--sketcher-name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--result-path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--save-path', type=str, default='./evaluation')
    parser.add_argument('--latex', action='store_true', help='Output results in LaTeX table format')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to files')
    args = parser.parse_args()
    
    if args.sketcher_name:
        sketcher_names = [args.sketcher_name]
    else:
        config_path = './configs/models'
        sketcher_names = [os.path.splitext(f)[0] for f in os.listdir(config_path) 
                         if os.path.isfile(os.path.join(config_path, f))]
    
    all_results = {}
    for sketcher_name in sketcher_names:
        try:
            prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
            filename = f'{prefix}{args.dataset_name}_{args.split}_{sketcher_name}.json'
            result_file = os.path.join(args.result_path, filename)
            
            with open(result_file, 'r') as f:
                samples = json.load(f)

            all_results[sketcher_name] = LogicEvaluator.evaluate_sample_groups(samples, with_string=False)
            
        except Exception as e:
            print(f"Error processing {sketcher_name}: {e}")
            continue
    
    # Print results   
    if args.latex:
        print("\nLaTeX Table:")
        print(ResultFormatter.format_latex(all_results))
    else:
        df = ResultFormatter.create_dataframe(all_results)
        print("\nResults:")
        print("="*80)
        print(df)
        print("="*80)
    
    # Save results unless explicitly disabled
    if not args.no_save:
        csv_path, latex_path, json_path = save_results(all_results, args)
        print("\nResults saved to:")
        print(f"CSV: {csv_path}")
        print(f"LaTeX: {latex_path}")
        print(f"JSON: {json_path}")

if __name__ == "__main__":
    main()