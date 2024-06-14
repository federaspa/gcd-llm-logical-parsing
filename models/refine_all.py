import os
import argparse

from self_refinement import SelfRefinementEngine, GrammarConstrainedModel

sketchers = ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']
refiners = [None]
load_dirs = ['outputs_1']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument('--n_gpu_layers', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for load_dir in load_dirs:
        for sketcher in sketchers:
            for refiner_path in refiners:
                
                args.sketcher_name = sketcher
                args.refiner_path = refiner_path
                # args.predicates_path = os.path.join(output_dir, 'logic_predicates')
                # args.programs_path = os.path.join(output_dir, 'logic_programs')
                args.load_dir = load_dir
                
                starting_round = 1
                
                
                if args.refiner_path:
                    
                    # print(f"Using refiner model from {args.refiner_path}.")
                    
                    refiner=GrammarConstrainedModel(
                        refiner_path=args.refiner,
                        n_gpu_layers=args.n_gpu_layers,
                    )
                else:
                    
                    # print(f"Using OpenAI model {args.sketcher_name} as refiner.")
                    refiner=None
                
                for round in range(starting_round, args.maximum_rounds+1):
                    print(f"Round {round} self-refinement")
                    engine = SelfRefinementEngine(args, round, refiner = refiner)
                    engine.single_round_self_refinement()