import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import argparse
from sklearn.metrics import  precision_recall_fscore_support

class Statistics:
    def __init__(self):
        self.stats = {}

    def add_stat(self, name, value, prefix='', suffix=''):
        if prefix not in self.stats:
            self.stats[prefix] = {}
        if suffix not in self.stats[prefix]:
            self.stats[prefix][suffix] = {}
        if name not in self.stats[prefix][suffix]:
            self.stats[prefix][suffix][name] = []
        self.stats[prefix][suffix][name].append(value)

    def get_stat(self, name, prefix='', suffix=''):
        if prefix not in self.stats or suffix not in self.stats[prefix] or name not in self.stats[prefix][suffix]:
            return None
        return self.stats[prefix][suffix][name]

    def average_stat(self, name, prefix='', suffix=''):
        values = self.get_stat(name, prefix, suffix)
        if values is None:
            return None
        return sum(values) / len(values)

average_all_stats = Statistics()
average_executable_stats = Statistics()
average_rates = Statistics()

text_to_index = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'N/A': 8
}

index_to_text = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'N/A'
}
def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')

    if answer_str.startswith(':'):
       return answer_str.replace(':', '').replace('.', '').strip()
    return 'N/A'

def evaluate_metrics(QA_results, average='weighted'):
    
    predictions = [text_to_index[get_choice(sample['predicted_answer'])] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    return precision_recall_fscore_support(gold_answers, predictions, average=average, zero_division=np.nan)

def get_backup_answers(samples, backup):
    
    non_executable_samples = [sample for sample in samples if sample['flag'] != 'success']
    
    for sample in non_executable_samples:
        sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
        
    return samples

def evaluation(result_file, backup_file, use_backup):
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
    
    executable_rate = len(executable_samples)/len(all_samples)
    parsing_errors_rate = len(parsing_errors)/len(all_samples)
    execution_errors_rate = len(execution_errors)/len(all_samples)
    rates = [executable_rate, parsing_errors_rate, execution_errors_rate]


    return all_stats, executable_stats, class_stats, rates

def get_res(metric):
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    
    res = str(mean) + '^{\\pm' + str(std) + '}'
    
    return res

def print_results(prefix, suffix):  
    
    def print_stat(stat, metrics=['precision', 'recall', 'f1']):
        for metric in metrics:
            all_stats = stat.get_stat(metric, prefix, suffix)
            print(metric.capitalize() + ':', get_res(all_stats))
            print()
      
    print('All Stats:')
    print_stat(average_all_stats)

    print('Executable Stats:')
    print_stat(average_executable_stats)
    
    print('Rates:')
    print_stat(average_rates, ['executable_rate', 'parsing_errors_rate', 'execution_errors_rate'])

def update_stats(all_stats, executable_stats, rates, prefix, suffix):
    average_all_stats.add_stat('precision', all_stats[0], prefix, suffix)
    average_all_stats.add_stat('recall', all_stats[1], prefix, suffix)
    average_all_stats.add_stat('f1', all_stats[2], prefix, suffix)
    # all_stats.add_stat('support', all_stats[3], suffix)
    
    average_executable_stats.add_stat('precision', executable_stats[0], prefix, suffix)
    average_executable_stats.add_stat('recall', executable_stats[1], prefix, suffix)
    average_executable_stats.add_stat('f1', executable_stats[2], prefix, suffix)
    # executable_stats.add_stat('support', executable_stats[3], suffix)
    
    average_rates.add_stat('executable_rate', rates[0], prefix, suffix)
    average_rates.add_stat('parsing_errors_rate', rates[1], prefix, suffix)
    average_rates.add_stat('execution_errors_rate', rates[2], prefix, suffix)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument('--self_refine_round', type=int, default=0)
    # parser.add_argument("--sketcher_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static', 'logiclm'], default='dynamic')
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--refiner_name", type=str, default=None)
    parser.add_argument("--use_backup", action='store_true' ,default=False)
    parser.add_argument("--gcd", action='store_true')
    parser.add_argument("--make_plot", action='store_true')
    parser.add_argument("--sketcher_list", type=str, default=['gpt-3.5-turbo', 'gpt-4o'])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
        
    if args.load_dir:
        load_dirs = [args.load_dir]
    else:
        load_dirs = ['outputs_1', 'outputs_2', 'outputs_3']
            
    args.sketcher_list = args.sketcher_list.split(',')        
    
    
    for load_dir in load_dirs:
        
        for prefix in 'gcd', 'no_gcd':
            
            if args.refiner_name is None:
                prefix = ''
        
            for suffix in '', '-finetune':
            
                if args.refiner_name is None:
                    result_path = os.path.join(load_dir, 'logic_inference')
                    # raise NotImplementedError('Please provide a refiner name')
                else:
                    # gcd_folder = 'gcd' if args.gcd else 'no_gcd'
                    result_path = os.path.join(load_dir, 'logic_inference', prefix, args.refiner_name + suffix)
                    
                # suffix = suffix.replace('-', '')
                
                for sketcher in args.sketcher_list:
                    
                    try:
                        args.sketcher_name = sketcher
                        
                        new_suffix = sketcher + suffix
                    
                        backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.sketcher_name}.json')
                        
                        if args.self_refine_round > 0:
                            result_file = os.path.join(result_path, f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')
                        else:
                            result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')

                        all_stats, executable_stats, class_stats, rates = evaluation(result_file, backup_file, use_backup=args.use_backup)
                        
                        
                        update_stats(all_stats, executable_stats, rates, prefix, new_suffix)
                    
                    except FileNotFoundError:
                        print(f'File not found: {result_file}')
                    
                if args.refiner_name is None:
                    break
                
            if args.refiner_name is None:
                break

    
    if args.refiner_name is not None:
        for prefix in 'gcd', 'no_gcd':
            print('GCD' if prefix == 'gcd' else 'No GCD')
            print('--------------------------------')
            for suffix in '', '-finetune':
                for sketcher in args.sketcher_list:
                    args.sketcher_name = sketcher
                    new_suffix = sketcher + suffix
                    title = 'Finetune' if 'finetune' in new_suffix else 'No finetune'
                    title += ' ' + sketcher
                    print(title)
                    print()
                    print('-------------------------------------')
                    print_results(prefix, new_suffix)
                    print('-------------------------------------')
          
                    
    else:
        for sketcher in args.sketcher_list:
            args.sketcher_name = sketcher
            new_suffix = sketcher
            title = sketcher
            print(title)
            print()
            print('-------------------------------------')
            print_results('', new_suffix)
            print('--------------------------------')
            
        
    if args.make_plot:
        # make a plot to compare average performance with and without finetuning and with and without GCD for each sketcher
        n_groups = 4
        means_finetune = [average_all_stats.average_stat('f1', 'gcd', 'gpt-3.5-turbo-finetune'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-3.5-turbo-finetune'), average_all_stats.average_stat('f1', 'gcd', 'gpt-4o-finetune'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-4o-finetune')]
        means_no_finetune = [average_all_stats.average_stat('f1', 'gcd', 'gpt-3.5-turbo'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-3.5-turbo'), average_all_stats.average_stat('f1', 'gcd', 'gpt-4o'), average_all_stats.average_stat('f1', 'no_gcd', 'gpt-4o')]

        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
        
        rects1 = plt.bar(index, means_finetune, bar_width, alpha=opacity, color='b', label='Finetune')
        rects2 = plt.bar(index + bar_width, means_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune')
        
        plt.xlabel('Sketchers')
        plt.ylabel('F1 score')
        plt.title('F1 by sketcher')
        plt.xticks(index + bar_width, ('GPT-3.5-Turbo\nGCD', 'GPT-3.5-Turbo\nNo GCD', 'GPT-4o\nGCD', 'GPT-4o\nNo GCD'))
        plt.legend()
        
        plt.yscale('log')
        yticks = np.logspace(np.log10(min(means_finetune + means_no_finetune)), np.log10(max(means_finetune + means_no_finetune)), 10)
        plt.gca().yaxis.set_major_locator(ticker.FixedLocator(yticks))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x)))
            
        
        plt.tight_layout()
        plt.savefig('performance.png')
        
        # make a plot to compare average rates with and without finetuning and with and without GCD for each sketcher
        
        n_groups = 4
        means_finetune = [average_rates.average_stat('executable_rate', 'gcd', 'gpt-3.5-turbo-finetune'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-3.5-turbo-finetune'), average_rates.average_stat('executable_rate', 'gcd', 'gpt-4o-finetune'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-4o-finetune')]
        means_no_finetune = [average_rates.average_stat('executable_rate', 'gcd', 'gpt-3.5-turbo'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-3.5-turbo'), average_rates.average_stat('executable_rate', 'gcd', 'gpt-4o'), average_rates.average_stat('executable_rate', 'no_gcd', 'gpt-4o')]
        
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8
        
        rects1 = plt.bar(index, means_finetune, bar_width, alpha=opacity, color='b', label='Finetune')
        rects2 = plt.bar(index + bar_width, means_no_finetune, bar_width, alpha=opacity, color='r', label='No finetune')
        
        plt.xlabel('Sketchers')
        plt.ylabel('Executable Rate')
        plt.title('Executable Rate by sketcher')
        plt.xticks(index + bar_width, ('GPT-3.5-Turbo\nGCD', 'GPT-3.5-Turbo\nNo GCD', 'GPT-4o\nGCD', 'GPT-4o\nNo GCD'))

        plt.yscale('log')
        yticks = np.logspace(np.log10(min(means_finetune + means_no_finetune)), np.log10(max(means_finetune + means_no_finetune)), 10)
        plt.gca().yaxis.set_major_locator(ticker.FixedLocator(yticks))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x)))


        plt.legend()
        
        plt.tight_layout()
        plt.savefig('executable_rate.png')