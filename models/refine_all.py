import os
import argparse
import sys

from self_refinement import SelfRefinementEngine, GrammarConstrainedModel

sketchers = [
    'gpt-3.5-turbo',
    # 'gpt-4-turbo',
    'gpt-4o'
]
refiners = [
    'GCD/llms/llama-2-7b.gguf',
    'GCD/llms/llama-2-7b-finetune.gguf',
    'GCD/llms/llama-2-13b.gguf',
    'GCD/llms/llama-2-13b-finetune.gguf',
    # 'GCD/llms/upstage-llama-2-70b-instruct-v2.Q6_K.gguf',
    'GCD/llms/llama-3-8b.gguf',
    'GCD/llms/llama-3-8b-finetune.gguf',
    # 'GCD/llms/Meta-Llama-3-70B-Instruct-v2.Q6_K-00001-of-00002.gguf',
    # 'GCD/llms/mistral-7b.gguf',
    # 'GCD/llms/mixtral-8x7b-instruct-v0.1.Q6_K.gguf',
    # 'GCD/llms/Mixtral_8x22B/Q6_K-00001-of-00004.gguf',
    # None
]
load_dirs = [
    'outputs/outputs_1',
    'outputs/outputs_2',
    'outputs/outputs_3'
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument('--n_gpu_layers', type=int)
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.gcd:
        print('No GCD')

    for load_dir in load_dirs:
        for sketcher in sketchers:
            for refiner_path in refiners:

                print(f'sketcher: {sketcher}, refiner {refiner_path}, load_dir: {load_dir}')

                try:
                    args.sketcher_name = sketcher
                    args.refiner_path = refiner_path
                    args.load_dir = load_dir

                    starting_round = 1

                    gcd_dir = 'gcd' if args.gcd else 'no_gcd'
                    refiner_name = refiner_path.split('/')[-1].split('.')[0] if refiner_path else sketcher
                    
                    all_exist = True
                    for current_round in range(starting_round, args.maximum_rounds+1):
                        save_path = os.path.join(load_dir, 'logic_programs', gcd_dir, refiner_path, f'self-refine-{current_round}_{args.dataset_name}_{args.split}_{sketcher}_{args.prompt_mode}.json')
                        
                        if not os.path.exists(save_path):
                            all_exist = False
                            break
                    
                    if all_exist:
                        print('All files exist')
                        continue
                    
                    if args.refiner_path:

                        print(f"Loading refiner {args.refiner_path}")
                        refiner=GrammarConstrainedModel(
                            refiner_path=args.refiner_path,
                            n_gpu_layers=args.n_gpu_layers,
                        )
                    else:

                        refiner=None

                    for round in range(starting_round, args.maximum_rounds+1):
                        print(f"Round {round} self-refinement")
                        engine = SelfRefinementEngine(args, round, refiner = refiner)
                        engine.single_round_self_refinement()
                except KeyboardInterrupt:
                    sys.exit(0)
                except Exception as e:
                    print(e)