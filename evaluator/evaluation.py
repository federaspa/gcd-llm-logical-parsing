# main.py
import argparse
import json
import os
from utils.metrics import MetricsCalculator
from utils.formatters import ResultFormatter
from utils.utils import get_models

num_samples = {
    'FOLIO': 204,
    'GSM8K_symbolic': 1000,
    'ProofWriter': 340
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-path', type=str, required=True)
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--models-path', type=str, default='/home/fraspanti/LLMs')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--save-df', action='store_true', help='Save results as dataframe')
    parser.add_argument('--latex', action='store_true', help='Output results in LaTeX table format')
    parser.add_argument('--latex-split', action='store_true', help='Output results in two LaTeX tables')
    args = parser.parse_args()
    
    model_names = [args.model_name] if args.model_name else get_models(args.models_path)
        
    if args.dataset_name:
        dataset_names = [args.dataset_name]
    else:
        dataset_names = ['FOLIO', 'GSM8K_symbolic']
    
    all_results = {model:{} for model in model_names}
    shot_numbers = ["0shots", "2shots", "5shots"]
    for model_name in model_names:
        for dataset_name in dataset_names:
            model_results = {}
            for shot in shot_numbers:
                try:
                    prefix = f'self-refine-{args.self_refine_round}_' if args.self_refine_round > 0 else ''
                    filename = f'{prefix}{dataset_name}_{args.split}_{model_name}.json'
                    result_file = os.path.join(args.result_path, f"{shot}", 'logic_inference', filename)
                    
                    with open(result_file, 'r') as f:
                        samples = json.load(f)
                        
                        
                    model_results[shot] = MetricsCalculator.evaluate_sample_groups(samples, num_samples[dataset_name])
                
                
                except FileNotFoundError:
                    model_results[shot] = MetricsCalculator.evaluate_sample_groups([])
                    # print(f"File not found: {filename}")
                    
                except Exception as e:
                    model_results[shot] = MetricsCalculator.evaluate_sample_groups([])
                    print(f"Error processing {model_name}: {e}")
                    
            all_results[model_name][dataset_name] = model_results
            
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
        else:
            df.to_csv(output_file, index=False)

    # Print results   
    if args.latex:
        print(ResultFormatter.format_latex(all_results))
        
    elif args.latex_split:
        print(ResultFormatter.format_latex_split(all_results))


if __name__ == "__main__":
    main()