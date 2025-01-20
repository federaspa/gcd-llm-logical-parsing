# main.py
import argparse
import json
import os
from core.logic_evaluator import LogicEvaluator
from utils.formatters import ResultFormatter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-name', type=str)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--result-path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--latex', action='store_true', help='Output results in LaTeX table format')
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

            all_results[sketcher_name] = LogicEvaluator.evaluate_sample_groups(samples, True)
            
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

if __name__ == "__main__":
    main()