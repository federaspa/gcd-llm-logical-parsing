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
    
def evaluation(result_files: list[str, str, str, str], thingy:list) -> Tuple:
    
    raw_file = result_files[0]
    grammar_file = result_files[1]
    
    raw_file_prog = result_files[0].replace('logic_inference', 'logic_programs')
    grammar_file_prog = result_files[1].replace('logic_inference', 'logic_programs')

    
    with open(raw_file, 'r') as f:
        raw_samples = json.load(f)
        
    with open(grammar_file, 'r') as f:
        grammar_samples = json.load(f)
        
    with open(grammar_file_prog, 'r') as f:
        grammar_samples_prog = json.load(f)
        
    with open(raw_file_prog, 'r') as f:
        raw_samples_prog = json.load(f)
        
    
    # assert len(raw_samples) == len(grammar_samples), f'Len mismatch samples in {raw_file} vs {grammar_file}: {len(raw_samples)} != {len(grammar_samples)}'
        
    unique_ids = set([sample['id'] for sample in raw_samples]) - set([sample['id'] for sample in grammar_samples])
    
    # remove samples with unique ids from raw_samples
    raw_samples = [sample for sample in raw_samples if sample['id'] not in unique_ids]

    
    grammar_fixed = []
            
    thingy_key = '_'.join(grammar_file.split('/')[-3:-1])
    thingy_key = thingy_key + '_3.5' if '3.5' in grammar_file else thingy_key + '_4'
    
    
    if not thingy_key in thingy:
        thingy[thingy_key] = []
            
    tot = 0
    errors = 0
            
    for raw_sample, grammar_sample in zip(raw_samples, grammar_samples):
        if raw_sample['flag'] != 'success' and grammar_sample['flag'] == 'success':
            
            tot += 1
            grammar_fixed.append(grammar_sample)
            
            # fix but no improvement
            if (raw_sample['id'] not in [thing['id'] for thing in thingy[thingy_key]]) and (grammar_sample['answer'] != grammar_sample['predicted_answer']) and (raw_sample['flag'] == 'parsing error'):
                
                raw_prog = [sample for sample in raw_samples_prog if sample['id'] == raw_sample['id']][0]["raw_logic_programs"]
                gram_prog = [sample for sample in grammar_samples_prog if sample['id'] == raw_sample['id']][0]["raw_logic_programs"]
                diff = []
                
                
                try:
                    raw_rules = raw_prog["First-Order-Logic Rules"]
                    gram_rules = gram_prog["First-Order-Logic Rules"]
                    
                    raw_rules = raw_rules.split('\n') if type(raw_rules) == str else raw_rules
                    gram_rules = gram_rules.split('\n') if type(gram_rules) == str else gram_rules
                    
                    assert len(raw_rules) == len(gram_rules), 'Len mismatch rules'
                    
                    raw_question = raw_prog["First-Order-Logic Question"]
                    gram_question = gram_prog["First-Order-Logic Question"]
                    
                    raw_question = raw_question.split('\n') if type(raw_question) == str else raw_question
                    gram_question = gram_question.split('\n') if type(gram_question) == str else gram_question
                    
                    assert len(raw_question) == len(gram_question), 'Len mismatch question'
                                        
                    for r, g in zip(raw_rules, gram_rules):
                        if SequenceMatcher(None, r, g).ratio() < 0.95:
                          diff.append({
                              'raw': r,
                              'gram': g
                          })
                          
                    for r, g in zip(raw_question, gram_question):
                        if SequenceMatcher(None, r, g).ratio() < 0.95:
                          diff.append({
                              'raw': r,
                              'gram': g
                          })
                            
                    # thingy[thingy_key].append({
                    #     'id': raw_sample['id'],
                    #     'diff': diff,
                    #     'flag': raw_sample['flag'],
                        
                    #     'answer': grammar_sample['answer'],
                    #     'grammar_answer': grammar_sample['predicted_answer']
                    # })
                except Exception as e:
                    print(raw_sample['id'], e)
                finally:
                    thingy[thingy_key].append({
                        'id': raw_sample['id'],
                        'raw_prog': raw_prog,
                        'gram_prog': gram_prog,
                        'diff': diff,
                        # 'flag': raw_sample['flag'],
                        'answer': grammar_sample['answer'],
                        'grammar_answer': grammar_sample['predicted_answer']
                    })
            
    return thingy

def get_load_dirs(args: Args) -> List[str]:
    return [args.load_dir] if args.load_dir else ['outputs/outputs_1', 'outputs/outputs_2', 'outputs/outputs_3']

def get_result_path(load_dir: str, refiner_name:str, prefix: str) -> str:
    if refiner_name in ['gpt-3.5-turbo', 'gpt-4o']:
        return os.path.join(load_dir, 'logic_inference', refiner_name)
    
    else:
        return os.path.join(load_dir, 'logic_inference', prefix, refiner_name)

def get_file_names(args: Args, sketcher: str, prompt_mode:str, result_path: str, self_refine_round) -> Tuple[str, str]:
            
    result_file_name = f'{"self-refine-" + str(self_refine_round) + "_" if self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{sketcher}_{prompt_mode}.json'
    result_file = os.path.join(result_path, result_file_name)
    
    return result_file


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

def process_data(args: Args) -> dict:
    load_dirs = get_load_dirs(args)
    thingy = {}
    
    refiners_list = args.refiners_list
    
    for refiner in refiners_list:  # iterate over all refiners
        
        for sketcher in args.sketcher_list:
            
            if sketcher == 'gpt-3.5-turbo' and refiner == 'gpt-4o':
                continue
            
            if sketcher == 'gpt-4o' and refiner == 'gpt-3.5-turbo':
                continue
        
            for load_dir in load_dirs:

                result_files = []
                
                result_files.append(get_file_names(args, sketcher, 'dynamic', get_result_path(load_dir, '', ''), 0))  
                result_files.append(get_file_names(args, sketcher, 'dynamic', get_result_path(load_dir, refiner, 'gcd',), args.self_refine_round))
                
                thingy = evaluation(result_files, thingy)
            
    
    
    return thingy

def main():
    args = parse_args()
    thingy = process_data(args)
    
    with open('fixed_errors.json', 'w') as f:
        json.dump(thingy, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()