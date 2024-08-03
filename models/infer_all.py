import os
import argparse

from logic_inference import LogicInferenceEngine

sketchers = [
    'gpt-3.5-turbo', 
    # 'gpt-4-turbo', 
    'gpt-4o'
    ]
refiners = [
    'llama-2-7b',
    'llama-2-7b-finetune',
    'llama-2-13b',
    'llama-2-13b-finetune',
    # 'llama-2-70b',
    'llama-3-8b',
    'llama-3-8b-finetune',
    'mistral-7b',
    # 'mixtral-8x7b',
    # 'mixtral_8x22B',
    # None
]
load_dirs = [
    'outputs_1', 
    'outputs_2', 
    'outputs_3'
    ]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument('--starting_round', type=int, default=3)
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for load_dir in load_dirs:
        for sketcher in sketchers:
            for refiner_path in refiners:

                print(f'sketcher: {sketcher}, refiner {refiner_path}, load_dir: {load_dir}, gcd: {args.gcd}')

                try:
                    refiner_name = refiner_path.split('/')[-1].split('.')[0] if refiner_path else sketcher
                    
                    gcd_folder = 'gcd' if args.gcd else 'no_gcd'
                    
                    args.programs_path = os.path.join(load_dir, 'logic_programs', gcd_folder, refiner_name)
                    
                    args.save_path = os.path.join(load_dir, 'logic_inference', gcd_folder, refiner_name)
                    
                    args.sketcher_name = sketcher
                    args.load_dir = load_dir

                    for round in range(args.starting_round, args.maximum_rounds + 1):
                        print(f"Round {round} self-refinement")
                        args.self_refine_round = round
                        engine = LogicInferenceEngine(args)
                        engine.inference_on_dataset()
                        
                except Exception as e:
                    print(e)