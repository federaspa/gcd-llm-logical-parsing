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
    'llama-2-13b',
    'upstage-llama-2-70b-instruct-v2',
    'Meta-Llama-3-8B-Instruct',
    'Meta-Llama-3-70B-Instruct-v2',
    'mistral-7b-instruct-v0',
    'mixtral-8x7b-instruct-v0',
    # 'Mixtral_8x22B',
            None
]
load_dirs = [
    # 'outputs_1', 
    'outputs_2', 
    # 'outputs_3'
    ]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for load_dir in load_dirs:
        for sketcher in sketchers:
            for refiner_path in refiners:

                print(f'sketcher: {sketcher}, refiner {refiner_path}, load_dir: {load_dir}')

                try:
                    refiner_name = refiner_path.split('/')[-1].split('.')[0] if refiner_path else sketcher
                    args.programs_path = os.path.join(load_dir, 'logic_programs', refiner_name)
                    
                    args.save_path = os.path.join(load_dir, 'logic_inference', refiner_name)
                    
                    args.sketcher_name = sketcher
                    args.load_dir = load_dir

                    for round in range(3, 4):
                        print(f"Round {round} self-refinement")
                        args.self_refine_round = round
                        engine = LogicInferenceEngine(args)
                        engine.inference_on_dataset()
                        
                except Exception as e:
                    print(e)