# main.py
import argparse
import json
import os
from utils.metrics import MetricsCalculator
from utils.formatters import ResultFormatter

num_samples = {
    'FOLIO': 204,
    'GSM8K_symbolic': 1000
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--save-df', action='store_true', help='Save results as dataframe')
    parser.add_argument('--latex', action='store_true', help='Output results in LaTeX table format')
    parser.add_argument('--latex-split', action='store_true', help='Output results in two LaTeX tables')
    args = parser.parse_args()
    
    if args.model_name:
        model_names = [args.model_name]
    else:
        config_path = './configs/models'
        model_names = [os.path.splitext(f)[0] for f in os.listdir(config_path) 
                         if os.path.isfile(os.path.join(config_path, f))]
    
    all_results = {}
    shot_numbers = [0, 2, 5]
    for model_name in model_names:
        model_results = {}
        for shot in shot_numbers:
            try:
                prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
                filename = f'{prefix}{args.dataset_name}_{args.split}_{model_name}.json'
                result_file = os.path.join(args.result_path, f"{shot}shots", 'logic_inference', filename)
                
                                
                with open(result_file, 'r') as f:
                    samples = json.load(f)
                    
                model_results[shot] = MetricsCalculator.evaluate_sample_groups(samples, num_samples[args.dataset_name])
            
            
            except FileNotFoundError:
                model_results[shot] = MetricsCalculator.evaluate_sample_groups([])
                
            except Exception as e:
                model_results[shot] = MetricsCalculator.evaluate_sample_groups([])
                print(f"Error processing {model_name}: {e}")
                
        all_results[model_name] = model_results
        
        
    if args.save_df:
        df = ResultFormatter.create_dataframe(all_results)
        
        output_file = os.path.join('evaluator', 'evaluation_results.csv')
        if os.path.exists(output_file):
            overwrite = input(f"{output_file} already exists. Do you want to overwrite it? (y/n): ")
            if overwrite.lower() != 'y':
                print("File not overwritten.")
                
            else:
                df.to_csv(output_file, index=False)
                print(f"Results saved to {output_file}")
                print("\nResults:")
                print("="*80)
                print(df)
                print("="*80)

    # Print results   
    if args.latex:
        print(ResultFormatter.format_latex(all_results))
        
    elif args.latex_split:
        print(ResultFormatter.format_latex_split(all_results))


if __name__ == "__main__":
    main()