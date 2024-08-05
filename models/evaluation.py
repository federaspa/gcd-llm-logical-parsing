import numpy as np
import pandas as pd
import json
import os
import traceback
import argparse
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, NamedTuple

class Statistics:
    def __init__(self):
        self.stats: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    def add_stat(self, name: str, value: float, prefix: str = '', suffix: str = ''):
        self.stats.setdefault(prefix, {}).setdefault(suffix, {}).setdefault(name, []).append(value)

    def get_stat(self, name: str, prefix: str = '', suffix: str = '') -> Optional[List[float]]:
        return self.stats.get(prefix, {}).get(suffix, {}).get(name)

    def average_stat(self, name: str, prefix: str = '', suffix: str = '') -> Optional[float]:
        values = self.get_stat(name, prefix, suffix)
        return np.mean(values) if values else None
    
class Args(NamedTuple):
    dataset_name: str
    self_refine_round: int
    split: str
    # prompt_mode: str
    load_dir: Optional[str]
    refiners_list: List[str]
    # use_backup: bool
    # gcd: bool
    # make_plot: bool
    sketcher_list: List[str]
    

TEXT_TO_INDEX = {c: i for i, c in enumerate('ABCDEFGH')}
TEXT_TO_INDEX['N/A'] = 8
INDEX_TO_TEXT = {v: k for k, v in TEXT_TO_INDEX.items()}

def get_choice(answer_str: str) -> str:
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for c in choices:
        if answer_str.startswith(f"{c}") or answer_str.startswith(f"{c}.") or answer_str.startswith(f"{c})"):
            return c
    return answer_str.lstrip(':').rstrip('.').strip() if answer_str.startswith(':') else 'N/A'

def evaluate_metrics(QA_results: List[Dict], average: str = 'weighted') -> Tuple:
    predictions = [TEXT_TO_INDEX[get_choice(sample['predicted_answer'])] for sample in QA_results]
    gold_answers = [TEXT_TO_INDEX[sample['answer']] for sample in QA_results]
    return precision_recall_fscore_support(gold_answers, predictions, average=average, zero_division=np.nan)

def get_backup_answers(samples: List[Dict], backup: Dict[str, Dict]) -> List[Dict]:
    for sample in samples:
        if sample['flag'] != 'success':
            sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
    return samples

def evaluation_baseline(result_file: str) -> Tuple:
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    all_stats = evaluate_metrics(all_samples)
    class_stats = evaluate_metrics(all_samples, average=None)

    return all_stats, class_stats

def evaluation(result_file: str, backup_file: str, use_backup: bool) -> Tuple:
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    if use_backup:
        with open(backup_file, 'r') as f:
            backup_samples = json.load(f)
        all_samples = get_backup_answers(all_samples, {sample['id']: sample for sample in backup_samples})

    executable_samples = [sample for sample in all_samples if sample['flag'] == 'success']
    parsing_errors = [sample for sample in all_samples if sample['flag'] == 'parsing error']
    execution_errors = [sample for sample in all_samples if sample['flag'] == 'execution error']

    all_stats = evaluate_metrics(all_samples)
    executable_stats = evaluate_metrics(executable_samples)
    class_stats = evaluate_metrics(all_samples, average=None)

    rates = [
        len(executable_samples) / len(all_samples),
        len(parsing_errors) / len(all_samples),
        len(execution_errors) / len(all_samples)
    ]

    return all_stats, executable_stats, class_stats, rates

def get_load_dirs(args: Args) -> List[str]:
    return [args.load_dir] if args.load_dir else ['outputs_1', 'outputs_2', 'outputs_3']

def get_result_path(load_dir: str, refiner_name:str, prefix: str, suffix: str) -> str:
    # if refiner_name is None:
    #     return os.path.join(load_dir, 'logic_inference')
    return os.path.join(load_dir, 'logic_inference', prefix, refiner_name + suffix)

def get_file_names(args: Args, sketcher: str, prompt_mode:str, result_path: str) -> Tuple[str, str]:
    backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{sketcher}.json')
    
    result_file_name = f'{"self-refine-" + str(args.self_refine_round) + "_" if args.self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{sketcher}_{prompt_mode}.json'
    result_file = os.path.join(result_path, result_file_name)
    
    return backup_file, result_file

def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default='dev')
    # parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static', 'logiclm'], default='dynamic')
    parser.add_argument("--load_dir", type=str, default=None)
    
    parser.add_argument("--sketcher_list", type=str, default='gpt-3.5-turbo,gpt-4o')
    
    parser.add_argument("--refiners_list", type=str, default='gpt-3.5-turbo,gpt-4o')
    parser.add_argument('--self_refine_round', type=int, default=0)

    args = parser.parse_args()
    
    return Args(
        dataset_name=args.dataset_name,
        self_refine_round=args.self_refine_round,
        split=args.split,
        # prompt_mode='',
        load_dir=args.load_dir,
        refiners_list=args.refiners_list.split(','),
        # use_backup=args.use_backup,
        # gcd=args.gcd,
        # make_plot=args.make_plot,
        sketcher_list=args.sketcher_list.split(',')
    )

def process_data(args: Args) -> pd.DataFrame:
    data = []
    load_dirs = get_load_dirs(args)
    
    refiners_list = args.refiners_list
    
    for refiner in refiners_list:  # iterate over all refiners
    
        prefixes = ['gcd', 'no_gcd'] if refiner not in ['gpt-3.5-turbo', 'gpt-4o'] else ['']
        
        for prompt_mode in ['dynamic', 'static']:
            # args.prompt_mode = prompt_mode
        
            for prefix in prefixes:
                suffixes = ['', '-finetune'] if refiner not in ['gpt-3.5-turbo', 'gpt-4o'] else ['']
                
                for suffix in suffixes:

                
                    for sketcher in args.sketcher_list:
                        
                        for use_backup in [True, False]:
                            
                            for load_dir in load_dirs:
                                
                                
                                result_path = get_result_path(load_dir, refiner, prefix, suffix)
                                backup_file, result_file = get_file_names(args, sketcher, prompt_mode, result_path)
                            
                                try:
                                    all_stats, executable_stats, _, rates = evaluation(result_file, backup_file, use_backup=use_backup)
                                    data.append({
                                        'sketcher': sketcher,
                                        'refiner': refiner,
                                        'load_dir': load_dir,
                                        'gcd': prefix == 'gcd',
                                        'finetune': suffix == '-finetune',
                                        'baseline': False,
                                        'use_backup': use_backup,
                                        'precision': all_stats[0],
                                        'recall': all_stats[1],
                                        'f1': all_stats[2],
                                        'executable_precision': executable_stats[0],
                                        'executable_recall': executable_stats[1],
                                        'executable_f1': executable_stats[2],
                                        'executable_rate': rates[0],
                                        'parsing_errors_rate': rates[1],
                                        'execution_errors_rate': rates[2],
                                    })
                                except FileNotFoundError:
                                    if prompt_mode == 'dynamic':                                    
                                        if refiner in ['gpt-3.5-turbo', 'gpt-4o']:
                                            if sketcher != refiner:
                                                continue
                                            
                                        print(f'File not found: {result_file}')
                                    
                                except KeyError as e:
                                    print(f'KeyError: {result_file}', e)
                    
    for load_dir in load_dirs:              
        # add baseline stats
        for sketcher in args.sketcher_list:
            logic_lm_file = os.path.join(load_dir, 'logic_inference', sketcher, f'{"self-refine-" + str(args.self_refine_round) + "_" if args.self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{sketcher}_logiclm.json')
            logic_lm_backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{sketcher}.json')
            
            for use_backup in [True, False]:
                try:
                    all_stats, executable_stats, _, rates = evaluation(logic_lm_file, logic_lm_backup_file, use_backup=use_backup)
                    data.append({
                        'sketcher': sketcher,
                        'refiner': sketcher,
                        'load_dir': load_dir,
                        'gcd': False,
                        'finetune': False,
                        'baseline': f'LogicLM (GPT-3.5-Turbo + GPT-3.5-Turbo)' if sketcher == 'gpt-3.5-turbo' else f'LogicLM (GPT-4o + GPT-4o)',
                        'use_backup': use_backup,
                        'precision': all_stats[0],
                        'recall': all_stats[1],
                        'f1': all_stats[2],
                        'executable_precision': executable_stats[0],
                        'executable_recall': executable_stats[1],
                        'executable_f1': executable_stats[2],
                        'executable_rate': rates[0],
                        'parsing_errors_rate': rates[1],
                        'execution_errors_rate': rates[2]
                    })
                except FileNotFoundError:
                    print(f'File not found: {logic_lm_file}')
                except KeyError as e:
                    print(f'KeyError: {logic_lm_file}', traceback.format_exc())

            for baseline in ['CoT']:
                cot_file = os.path.join('./baselines/results', f'{baseline}_{args.dataset_name}_{args.split}_{sketcher}.json')
                try:
                    all_stats, executable_stats = evaluation_baseline(cot_file)
                    data.append({
                        'sketcher': sketcher,
                        'refiner': sketcher,
                        'load_dir': load_dir,
                        'gcd': False,
                        'finetune': False,
                        'baseline': baseline + ('(GPT-3.5-Turbo)' if sketcher == 'gpt-3.5-turbo' else '(GPT-4o)'),
                        'use_backup': False,
                        'precision': all_stats[0],
                        'recall': all_stats[1],
                        'f1': all_stats[2],
                    })
                except FileNotFoundError:
                    print(f'File not found: {cot_file}')
                except KeyError as e:
                    print(f'KeyError: {cot_file}', traceback.format_exc())
    
    return pd.DataFrame(data)

def main():
    args = parse_args()
    df = process_data(args)
    df.to_csv('performance_data_folio.csv', index=False)

if __name__ == "__main__":
    main()