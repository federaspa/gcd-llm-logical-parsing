import numpy as np
import pandas as pd
import json
import os
import traceback
import argparse
from sklearn.metrics import precision_recall_fscore_support, f1_score
from typing import Dict, List, Tuple, Optional, NamedTuple
from copy import deepcopy
from difflib import SequenceMatcher
    
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
    save_path: str
    

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

def evaluation(result_files: list[str, str, str, str], backup_file:str) -> Tuple:
    
    raw_file = result_files[0]
    finetune_file = result_files[1]
    grammar_file = result_files[2]
    grammar_fine_file = result_files[3]

    
    with open(raw_file, 'r') as f:
        raw_samples = json.load(f)
        
    with open(finetune_file, 'r') as f:
        finetune_samples = json.load(f)
        
    with open(grammar_file, 'r') as f:
        grammar_samples = json.load(f)
        
    with open(grammar_fine_file, 'r') as f:
        grammar_fine_samples = json.load(f)
        
    assert len(raw_samples) == len(finetune_samples) == len(grammar_samples) == len(grammar_fine_samples), 'Len mismatch samples'
    
    
    fine_fixed = []
    grammar_fixed = []
    grammar_fine_fixed = []
        
    for raw_sample, fine_sample in zip(raw_samples, finetune_samples):
        if raw_sample['flag'] != 'success' and fine_sample['flag'] == 'success':
            fine_fixed.append(fine_sample)
            
    for raw_sample, grammar_sample in zip(raw_samples, grammar_samples):
        if raw_sample['flag'] != 'success' and grammar_sample['flag'] == 'success':
            
            grammar_fixed.append(grammar_sample)
            
    for raw_sample, grammar_fine_sample in zip(raw_samples, grammar_fine_samples):
        if raw_sample['flag'] != 'success' and grammar_fine_sample['flag'] == 'success':
            grammar_fine_fixed.append(grammar_fine_sample)
            
    
    raw = evaluate_metrics(raw_samples)[2], len(raw_samples)
    fine = evaluate_metrics(fine_fixed)[2], len(fine_fixed)
    grammar = evaluate_metrics(grammar_fixed)[2], len(grammar_fixed)
    grammar_fine = evaluate_metrics(grammar_fine_fixed)[2], len(grammar_fine_fixed)
    
    with open(backup_file, 'r') as f:
        backup_samples = json.load(f)
    backed_up_samples = get_backup_answers(raw_samples, {sample['id']: sample for sample in backup_samples})
    backed_up_samples = [sample for sample in backed_up_samples if sample['flag'] != 'success']  
    
    backup = evaluate_metrics(backed_up_samples)[2], len(backed_up_samples)
    
    random_samples = deepcopy(raw_samples)
    for random_samp in random_samples:
        random_samp['predicted_answer'] = INDEX_TO_TEXT[np.random.choice([0,1,2])]
        
    random = evaluate_metrics(random_samples)[2], len(random_samples)

    return raw, fine, grammar, grammar_fine, backup, random

def get_load_dirs(args: Args) -> List[str]:
    return [args.load_dir] if args.load_dir else ['outputs/outputs_1', 'outputs/outputs_2', 'outputs/outputs_3']

def get_result_path(load_dir: str, refiner_name:str, prefix: str, suffix: str) -> str:
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
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--sketcher_list", type=str, default='gpt-3.5-turbo,gpt-4o')
    parser.add_argument("--refiners_list", type=str, default='gpt-3.5-turbo,gpt-4o')
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--save_path", type=str, required=True)

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
        sketcher_list=args.sketcher_list.split(','),
        save_path=args.save_path
    )

def process_data(args: Args) -> pd.DataFrame:
    data = []
    load_dirs = get_load_dirs(args)
    
    refiners_list = args.refiners_list
    
    for refiner in refiners_list:  # iterate over all refiners
    
        prefixes = ['no_gcd', 'gcd'] if refiner not in ['gpt-3.5-turbo', 'gpt-4o'] else ['']
        suffixes = ['', '-finetune'] if refiner not in ['gpt-3.5-turbo', 'gpt-4o'] else ['']
            
        for sketcher in args.sketcher_list:
        
            for load_dir in load_dirs:
                
                result_paths = []
                result_files = []
                
                for prefix in prefixes:
                    for suffix in suffixes: 
                        result_paths.append(get_result_path(load_dir, refiner, prefix, suffix))
                
                for result_path in result_paths:
                    backup_file, result_file = get_file_names(args, sketcher, 'dynamic', result_path)
                    result_files.append(result_file)
                    
                raw, fine, grammar, grammar_fine, backup, random = evaluation(result_files, backup_file)
                
                data.append({
                    'sketcher': sketcher,
                    'refiner': refiner,
                    'load\\_dir': load_dir,
                    'raw\\_f1': raw[0],
                    'raw\\_num': raw[1],
                    'fine\\_f1': fine[0],
                    'fine\\_num': fine[1],
                    'grammar\\_f1': grammar[0],
                    'grammar\\_num': grammar[1],
                    'grammar\\_fine\\_f1': grammar_fine[0],
                    'grammar\\_fine\\_num': grammar_fine[1],
                    'backup\\_f1': backup[0],
                    'backup\\_num': backup[1],
                    'random\\_f1' : random[0],
                    'random\\_num' : random[1]
                })
    
    
    
    return pd.DataFrame(data)

def main():
    args = parse_args()
    df = process_data(args)
    df.to_csv(args.save_path, index=False)

if __name__ == "__main__":
    main()