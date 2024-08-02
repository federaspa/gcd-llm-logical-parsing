import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import argparse
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional

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
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    return f"{mean}^{{\\pm{std}}}"

def print_results(prefix: str, suffix: str, stats: Dict[str, Statistics]):
    for stat_name, stat_obj in stats.items():
        print(f"{stat_name}:")
        metrics = ['precision', 'recall', 'f1'] if stat_name != 'Rates' else ['executable_rate', 'parsing_errors_rate', 'execution_errors_rate']
        for metric in metrics:
            values = stat_obj.get_stat(metric, prefix, suffix)
            if values:
                print(f"{metric.capitalize()}: {get_res(values)}\n")

def update_stats(all_stats: Tuple, executable_stats: Tuple, rates: List[float], prefix: str, suffix: str, stats: Dict[str, Statistics]):
    for i, metric in enumerate(['precision', 'recall', 'f1']):
        stats['All Stats'].add_stat(metric, all_stats[i], prefix, suffix)
        stats['Executable Stats'].add_stat(metric, executable_stats[i], prefix, suffix)
    
    for i, rate_name in enumerate(['executable_rate', 'parsing_errors_rate', 'execution_errors_rate']):
        stats['Rates'].add_stat(rate_name, rates[i], prefix, suffix)

def parse_args():
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
    return parser.parse_args()

def main():
    args = parse_args()
    load_dirs = [args.load_dir] if args.load_dir else ['outputs_1', 'outputs_2', 'outputs_3']
    args.sketcher_list = args.sketcher_list.split(',')

    stats = {
        'All Stats': Statistics(),
        'Executable Stats': Statistics(),
        'Rates': Statistics()
    }

    for load_dir in load_dirs:
        for prefix in (['gcd', 'no_gcd'] if args.refiner_name else ['']):
            for suffix in ['', '-finetune']:
                if args.refiner_name is None:
                    result_path = os.path.join(load_dir, 'logic_inference')
                else:
                    result_path = os.path.join(load_dir, 'logic_inference', prefix, args.refiner_name + suffix)

                for sketcher in args.sketcher_list:
                    try:
                        args.sketcher_name = sketcher
                        new_suffix = sketcher + suffix
                        
                        backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.sketcher_name}.json')
                        
                        result_file = os.path.join(result_path, f'{"self-refine-" + str(args.self_refine_round) + "_" if args.self_refine_round > 0 else ""}{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')

                        all_stats, executable_stats, class_stats, rates = evaluation(result_file, backup_file, use_backup=args.use_backup)
                        
                        update_stats(all_stats, executable_stats, rates, prefix, new_suffix, stats)
                    
                    except FileNotFoundError:
                        print(f'File not found: {result_file}')
                
                if args.refiner_name is None:
                    break
            
            if args.refiner_name is None:
                break

    if args.refiner_name is not None:
        for prefix in ['gcd', 'no_gcd']:
            print('################################')
            print('GCD' if prefix == 'gcd' else 'No GCD')
            print('################################')
            for suffix in ['', '-finetune']:
                for sketcher in args.sketcher_list:
                    new_suffix = sketcher + suffix
                    title = f"{'Finetune' if 'finetune' in new_suffix else 'No finetune'} {sketcher}"
                    print(f"{title}\n")
                    print("-------------------------------------")
                    print_results(prefix, new_suffix, stats)
                    print("-------------------------------------")
    else:
        for sketcher in args.sketcher_list:
            print(f"{sketcher}\n")
            print("-------------------------------------")
            print_results('', sketcher, stats)
            print("-------------------------------------")

    if args.make_plot:
        make_plots(stats)

def make_plots(stats):
# make a plot to compare average performance with and without finetuning and with and without GCD for each sketcher
    # n_groups = 4
    # means_finetune = [average_all_stats.average_stat('f1', 'gcd', 'gpt-3.5-turbo-finetune'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-3.5-turbo-finetune'), average_all_stats.average_stat('f1', 'gcd', 'gpt-4o-finetune'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-4o-finetune')]
    # means_no_finetune = [average_all_stats.average_stat('f1', 'gcd', 'gpt-3.5-turbo'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-3.5-turbo'), average_all_stats.average_stat('f1', 'gcd', 'gpt-4o'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-4o')]

    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 0.8
    
    # rects1 = plt.bar(index, means_finetune, bar_width, alpha=opacity, color='b', label='Finetune')
    # rects2 = plt.bar(index + bar_width, means_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune')
    
    # plt.xlabel('Sketchers')
    # plt.ylabel('F1 score')
    # plt.title('F1 by sketcher')
    # plt.xticks(index + bar_width, ('GPT-3.5-Turbo\nGCD', 'GPT-3.5-Turbo\nNo GCD', 'GPT-4o\nGCD', 'GPT-4o\nNo GCD'))
    # plt.legend()
    
    # plt.yscale('log')
    # yticks = np.logspace(np.log10(min(means_finetune + means_no_finetune)), np.log10(max(means_finetune + means_no_finetune)), 10)
    # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(yticks))
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x)))
        
    
    # plt.tight_layout()
    # plt.savefig('performance.png')
    
    # # make a plot to compare average rates with and without finetuning and with and without GCD for each sketcher
    
    # n_groups = 4
    # means_finetune = [average_rates.average_stat('executable_rate', 'gcd', 'gpt-3.5-turbo-finetune'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-3.5-turbo-finetune'), average_rates.average_stat('executable_rate', 'gcd', 'gpt-4o-finetune'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-4o-finetune')]
    # means_no_finetune = [average_rates.average_stat('executable_rate', 'gcd', 'gpt-3.5-turbo'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-3.5-turbo'), average_rates.average_stat('executable_rate', 'gcd', 'gpt-4o'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-4o')]
    
    # fig, ax = plt.subplots()
    # index = np.arange(n_groups)
    # bar_width = 0.35
    # opacity = 0.8
    
    # rects1 = plt.bar(index, means_finetune, bar_width, alpha=opacity, color='b', label='Finetune')
    # rects2 = plt.bar(index + bar_width, means_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune')
    
    # plt.xlabel('Sketchers')
    # plt.ylabel('Executable Rate')
    # plt.title('Executable Rate by sketcher')
    # plt.xticks(index + bar_width, ('GPT-3.5-Turbo\nGCD', 'GPT-3.5-Turbo\nNo GCD', 'GPT-4o\nGCD', 'GPT-4o\nNo GCD'))

    # plt.yscale('log')
    # yticks = np.logspace(np.log10(min(means_finetune + means_no_finetune)), np.log10(max(means_finetune + means_no_finetune)), 10)
    # plt.gca().yaxis.set_major_locator(ticker.FixedLocator(yticks))
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x)))


    # plt.legend()
    
    # plt.tight_layout()
    # plt.savefig('executable_rate.png')
    pass

if __name__ == "__main__":
    main()