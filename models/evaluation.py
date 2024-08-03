import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
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

def get_res(metric: List[float]) -> str:
    metric = [m*100 for m in metric]
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    return f"{mean}^{{\\pm{std}}}%"

def update_stats(all_stats: Tuple, executable_stats: Tuple, rates: List[float], prefix: str, suffix: str, stats: Dict[str, Statistics]):
    for i, metric in enumerate(['precision', 'recall', 'f1']):
        stats['All Stats'].add_stat(metric, all_stats[i], prefix, suffix)
        stats['Executable Stats'].add_stat(metric, executable_stats[i], prefix, suffix)
    
    for i, rate_name in enumerate(['executable_rate', 'parsing_errors_rate', 'execution_errors_rate']):
        stats['Rates'].add_stat(rate_name, rates[i], prefix, suffix)

class Args(NamedTuple):
    dataset_name: str
    self_refine_round: int
    split: str
    prompt_mode: str
    load_dir: Optional[str]
    refiner_name: Optional[str]
    use_backup: bool
    gcd: bool
    make_plot: bool
    sketcher_list: List[str]

def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static', 'logiclm'], default='dynamic')
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--refiner_name", type=str, default=None)
    parser.add_argument("--use_backup", action='store_true', default=False)
    parser.add_argument("--gcd", action='store_true')
    parser.add_argument("--make_plot", action='store_true')
    parser.add_argument("--sketcher_list", type=str, default='gpt-3.5-turbo,gpt-4o')
    
    args = parser.parse_args()
    return Args(
        dataset_name=args.dataset_name,
        self_refine_round=args.self_refine_round,
        split=args.split,
        prompt_mode=args.prompt_mode,
        load_dir=args.load_dir,
        refiner_name=args.refiner_name,
        use_backup=args.use_backup,
        gcd=args.gcd,
        make_plot=args.make_plot,
        sketcher_list=args.sketcher_list.split(',')
    )

def get_load_dirs(args: Args) -> List[str]:
    return [args.load_dir] if args.load_dir else ['outputs_1', 'outputs_2', 'outputs_3']

def get_result_path(load_dir: str, args: Args, prefix: str, suffix: str) -> str:
    if args.refiner_name is None:
        return os.path.join(load_dir, 'logic_inference')
    return os.path.join(load_dir, 'logic_inference', prefix, args.refiner_name + suffix)

def get_file_names(args: Args, sketcher: str, result_path: str) -> Tuple[str, str]:
    backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{sketcher}.json')
    
    result_file_name = f'{"self-refine-" + str(args.self_refine_round) + "_" if args.self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{sketcher}_{args.prompt_mode}.json'
    result_file = os.path.join(result_path, result_file_name)
    
    return backup_file, result_file

def process_sketcher(args: Args, stats: Dict[str, Statistics], prefix: str, suffix: str, sketcher: str, refiner:str, result_path: str):
    backup_file, result_file = get_file_names(args, sketcher, result_path)
    
    try:
        all_stats, executable_stats, _, rates = evaluation(result_file, backup_file, use_backup=args.use_backup)
        new_suffix = sketcher + '_' + refiner + suffix
        update_stats(all_stats, executable_stats, rates, prefix, new_suffix, stats)
    except FileNotFoundError:
        print(f'File not found: {result_file}')
        
def add_baseline_stats(args: Args, load_dir: str, stats: Dict[str, Statistics], sketcher: str):
    logic_lm_file = os.path.join(load_dir, 'logic_inference', sketcher, f'{"self-refine-" + str(args.self_refine_round) + "_" if args.self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{sketcher}_logiclm.json')
    try:
        all_stats, executable_stats, _, rates = evaluation(logic_lm_file, logic_lm_file, use_backup=args.use_backup)
        update_stats(all_stats, executable_stats, rates, '', sketcher + '-logiclm', stats)
    except FileNotFoundError:
        print(f'File not found: {logic_lm_file}')
    

def process_data(args: Args) -> Dict[str, Statistics]:
    stats = {
        'All Stats': Statistics(),
        'Executable Stats': Statistics(),
        'Rates': Statistics()
    }

    load_dirs = get_load_dirs(args)
    
    for load_dir in load_dirs:
        if args.refiner_name not in ['gpt-3.5-turbo', 'gpt-4o', None]:
            prefixes = ['gcd', 'no_gcd'] 
        else: 
            prefixes = ['']
            
        for prefix in prefixes:
            suffixes = ['', '-finetune']
            for suffix in suffixes:
                result_path = get_result_path(load_dir, args, prefix, suffix)
                
                for sketcher in args.sketcher_list:
                    process_sketcher(args, stats, prefix, suffix, sketcher, args.refiner_name, result_path)
                    add_baseline_stats(args, load_dir, stats, sketcher)
                
                if args.refiner_name in ['gpt-3.5-turbo', 'gpt-4o', None]:
                    break
            
            if args.refiner_name in ['gpt-3.5-turbo', 'gpt-4o', None]:
                break
        
        # # print stats for each load_dir
        # print_results_for_no_refiner(stats, args) if args.refiner_name in ['gpt-3.5-turbo', 'gpt-4o', None] else print_results_for_refiner(stats, args)

    return stats

def print_results(prefix: str, suffix: str, stats: Dict[str, Statistics]):
    for stat_name, stat_obj in stats.items():
        print(f"{stat_name}:")
        print()
        metrics = ['precision', 'recall', 'f1'] if stat_name != 'Rates' else ['executable_rate', 'parsing_errors_rate', 'execution_errors_rate']
        for metric in metrics:
            values = stat_obj.get_stat(metric, prefix, suffix)
            if values:
                print(f"{metric.capitalize()}: {get_res(values)}")
        print()
        

def print_results_for_refiner(stats: Dict[str, Statistics], args: Args):
    for prefix in ['gcd', 'no_gcd']:
        print('################################')
        print('GCD' if prefix == 'gcd' else 'No GCD')
        print('################################')
        print("-------------------------------------")
        for suffix in ['', '-finetune']:
            for sketcher in args.sketcher_list:
                new_suffix = sketcher + '_' + args.refiner_name + suffix
                title = f"{sketcher.capitalize()} + {args.refiner_name.capitalize()}{suffix}"
                print(f"{title}")
                print("-------------------------------------")
                print_results(prefix, new_suffix, stats)
                print("-------------------------------------")

def print_results_for_no_refiner(stats: Dict[str, Statistics], args: Args):
    for sketcher in args.sketcher_list:
        print("-------------------------------------")
        print(f"{sketcher}")
        print("-------------------------------------")
        print_results('', sketcher, stats)
        print("-------------------------------------")

def print_baseline_results(stats: Dict[str, Statistics], args: Args):
    print('################################')
    print('Baseline')
    print('################################')
    print("-------------------------------------")
    for sketcher in args.sketcher_list:
        print(f"{sketcher.capitalize()} + LogicLM")
        print("-------------------------------------")
        print_results('', sketcher + '-logiclm', stats)
        print("-------------------------------------")

def main():
    args = parse_args()
    stats = process_data(args)

    print_baseline_results(stats, args)
    
    if args.refiner_name in ['gpt-3.5-turbo', 'gpt-4o', None]:
        print_results_for_no_refiner(stats, args)
    else:
        print_results_for_refiner(stats, args)
        
    if args.make_plot:
        make_plots(stats, args.sketcher_list, args.refiner_name)


def make_plots(stats, sketchers, refiner):
    # make a plot to compare average performance with and without finetuning and with and without GCD for each sketcher
    all_stats = stats['All Stats']
    rates = stats['Rates']

    n_groups = 8
    means_finetune = []
    means_finetune_gcd = []
    means_no_finetune = []
    means_no_finetune_gcd = []
    rates_finetune = []
    rates_finetune_gcd = []
    rates_no_finetune = []
    rates_no_finetune_gcd = []

    for sketcher in sketchers:
        finetune_key = sketcher + '_' + refiner + '-finetune'
        no_finetune_key = sketcher + '_' + refiner
        
        means_finetune.append(all_stats.average_stat('f1', 'no_gcd', finetune_key)*100)
        means_finetune_gcd.append(all_stats.average_stat('f1', 'gcd', finetune_key)*100)
        
        means_no_finetune.append(all_stats.average_stat('f1', 'no_gcd', no_finetune_key)*100)
        means_no_finetune_gcd.append(all_stats.average_stat('f1', 'gcd', no_finetune_key)*100)
        
        rates_finetune.append(rates.average_stat('executable_rate', 'no_gcd', finetune_key)*100)
        rates_finetune_gcd.append(rates.average_stat('executable_rate', 'gcd', finetune_key)*100)
        
        rates_no_finetune.append(rates.average_stat('executable_rate', 'no_gcd', no_finetune_key)*100)
        rates_no_finetune_gcd.append(rates.average_stat('executable_rate', 'gcd', no_finetune_key)*100)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    index = np.arange(n_groups // 4)
    bar_width = 0.2
    opacity = 0.8
    
    ax1.bar(index, means_finetune, bar_width, alpha=opacity, color='b', label='Finetune, No GCD')
    ax1.bar(index + bar_width, means_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune, No GCD')
    ax1.bar(index + 2*bar_width, means_finetune_gcd, bar_width, alpha=opacity, color='g', label='Finetune, GCD')
    ax1.bar(index + 3*bar_width, means_no_finetune_gcd, bar_width, alpha=opacity, color='y', label='No finetune, GCD')
    ax1.set_xlabel('Sketchers')
    ax1.set_ylabel('F1 score (%)')
    ax1.set_title('Reasoning Performance')
    ax1.set_xticks(index + 2*bar_width)
    ax1.set_xticklabels(sketchers)
    ax1.legend()
    
    # set y-axis to log scale
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig('stats_plot.png')
    
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    index = np.arange(n_groups // 4)
    bar_width = 0.2
    opacity = 0.8
    
    ax2.bar(index, rates_finetune, bar_width, alpha=opacity, color='b', label='Finetune, No GCD')
    ax2.bar(index + bar_width, rates_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune, No GCD')
    ax2.bar(index + 2*bar_width, rates_finetune_gcd, bar_width, alpha=opacity, color='g', label='Finetune, GCD')
    ax2.bar(index + 3*bar_width, rates_no_finetune_gcd, bar_width, alpha=opacity, color='y', label='No finetune, GCD')
    ax2.set_xlabel('Sketchers')
    ax2.set_ylabel('F1 score')
    ax2.set_title('Executable Rate (%)')
    ax2.set_xticks(index + 2*bar_width)
    ax2.set_xticklabels(sketchers)
    ax2.legend()

    # set y-axis to log scale
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig('executable.png')


if __name__ == "__main__":
    main()