import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from sklearn.metrics import  precision_recall_fscore_support

average_all_stats = {
    'precision': 0,
    'recall': 0,
    'f1': 0,
    'support': 0,
    'precision-finetune': 0,
    'recall-finetune': 0,
    'f1-finetune': 0,
    'support-finetune': 0
}

average_executable_stats = {
    'precision': 0,
    'recall': 0,
    'f1': 0,
    'support': 0,
    'precision-finetune': 0,
    'recall-finetune': 0,
    'f1-finetune': 0,
    'support-finetune': 0
}

average_rates = {
    'executable_rate': 0,
    'parsing_errors_rate': 0,
    'execution_errors_rate': 0,
    'executable_rate-finetune': 0,
    'parsing_errors_rate-finetune': 0,
    'execution_errors_rate-finetune': 0
}

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

def evaluate_metrics(QA_results, average='weighted'):
    
    predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    # return f1_score(gold_answers, predictions, average=average, zero_division=np.nan), precision_score(gold_answers, predictions, average=average, zero_division=np.nan), recall_score(gold_answers, predictions, average=average, zero_division=np.nan)
    
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
    
    res = str(mean) + '^{\pm' + str(std) + '}'
    
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--sketcher_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--refiner_name", type=str, default=None)
    parser.add_argument("--use_backup", action='store_true' ,default=False)
    parser.add_argument("--gcd", action='store_true')
    
    args = parser.parse_args()
    return args

def print_results(suffix):
    print('All Stats:')
    print('Precision:', get_res([average_all_stats[f'precision{suffix}']]))
    print('Recall:', get_res([average_all_stats[f'recall{suffix}']]))
    print('F1:', get_res([average_all_stats[f'f1{suffix}']]))
    print()
    print('Executable Stats:')
    print('Precision:', get_res([average_executable_stats[f'precision{suffix}']]))
    print('Recall:', get_res([average_executable_stats[f'recall{suffix}']]))
    print('F1:', get_res([average_executable_stats[f'f1{suffix}']]))
    print()
    print('Rates:')
    print('Executable Rate:', get_res([average_rates[f'executable_rate{suffix}']]))
    print('Parsing Errors Rate:', get_res([average_rates[f'parsing_errors_rate{suffix}']]))
    print('Execution Errors Rate:', get_res([average_rates[f'execution_errors_rate{suffix}']]))


def update_stats(all_stats, executable_stats, rates, suffix):
    average_all_stats[f'precision{suffix}'] += all_stats[0]
    average_all_stats[f'recall{suffix}'] += all_stats[1]
    average_all_stats[f'f1{suffix}'] += all_stats[2]
    # average_all_stats['support'] += all_stats[3] if all_stats[3] else 0
    
    average_executable_stats[f'precision{suffix}'] += executable_stats[0]
    average_executable_stats[f'recall{suffix}'] += executable_stats[1]
    average_executable_stats[f'f1{suffix}'] += executable_stats[2]
    # average_executable_stats['support'] += executable_stats[3]
    
    average_rates[f'executable_rate{suffix}'] += rates[0]
    average_rates[f'parsing_errors_rate{suffix}'] += rates[1]
    average_rates[f'execution_errors_rate{suffix}'] += rates[2]
    
    return average_all_stats, average_executable_stats, average_rates

def average_stats(load_dirs):
    average_all_stats['precision'] /= len(load_dirs)
    average_all_stats['recall'] /= len(load_dirs)
    average_all_stats['f1'] /= len(load_dirs)
    average_all_stats['precision-finetune'] /= len(load_dirs)
    average_all_stats['recall-finetune'] /= len(load_dirs)
    average_all_stats['f1-finetune'] /= len(load_dirs)
    
    average_executable_stats['precision'] /= len(load_dirs)
    average_executable_stats['recall'] /= len(load_dirs)
    average_executable_stats['f1'] /= len(load_dirs)
    average_executable_stats['precision-finetune'] /= len(load_dirs)
    average_executable_stats['recall-finetune'] /= len(load_dirs)
    average_executable_stats['f1-finetune'] /= len(load_dirs)
    
    average_rates['executable_rate'] /= len(load_dirs)
    average_rates['parsing_errors_rate'] /= len(load_dirs)
    average_rates['execution_errors_rate'] /= len(load_dirs)
    average_rates['executable_rate-finetune'] /= len(load_dirs)
    average_rates['parsing_errors_rate-finetune'] /= len(load_dirs)
    average_rates['execution_errors_rate-finetune'] /= len(load_dirs)

if __name__ == "__main__":
    args = parse_args()
        
    if args.load_dir:
        load_dirs = [args.load_dir]
    else:
        load_dirs = ['outputs_1', 'outputs_2', 'outputs_3']
            
    for load_dir in load_dirs:
        
        for suffix in '', '-finetune':
        
            if args.refiner_name is None:
                # result_path = os.path.join(load_dir, 'logic_inference')
                raise NotImplementedError('Please provide a refiner name')
            else:
                gcd_folder = 'gcd' if args.gcd else 'no_gcd'
                result_path = os.path.join(load_dir, 'logic_inference', gcd_folder, args.refiner_name + suffix)
            
            backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.sketcher_name}.json')
            
            if args.self_refine_round > 0:
                result_file = os.path.join(result_path, f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')
            else:
                result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')

            all_stats, executable_stats, class_stats, rates = evaluation(result_file, backup_file, use_backup=args.use_backup)
            
            average_all_stats, average_executable_stats, average_rates = update_stats(all_stats, executable_stats, rates, suffix)

            
    average_stats(load_dirs)
    
    for suffix in '', '-finetune':
        print('No finetune' if suffix == '' else 'Finetune')
        print('-------------------------------------')
        print_results(suffix)
        print('######################################')
    
    # make a plot to compare performance with and without finetuning
    # data to plot
    n_groups = 4
    means_no_finetune = [average_all_stats['f1'], average_executable_stats['f1'], average_rates['executable_rate']]
    means_finetune = [average_all_stats['f1-finetune'], average_executable_stats['f1-finetune'], average_rates['executable_rate-finetune']]
    
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    
    rects1 = plt.bar(index, means_no_finetune, bar_width,
    alpha=opacity,
    color='b',
    label='No Finetune')
    
    rects2 = plt.bar(index + bar_width, means_finetune, bar_width,
    alpha=opacity,
    color='g',
    label='Finetune')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Scores by metrics and finetuning')
    plt.xticks(index + bar_width, ('All', 'Executable', 'Executable Rate'))
    plt.legend()
    
    plt.tight_layout()
    # save the plot
    plt.savefig('scores.png')

    