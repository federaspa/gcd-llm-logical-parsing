import numpy as np
import json
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support

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
        # sample['predicted_reasoning'] = backup[sample['id']]['predicted_reasoning']
        
    # print(f"Recovered {len(non_executable_samples)} samples from backup")
        
    return samples

def partial_evaluation(result_file, backup_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)
        
    with open(backup_file, 'r') as f:
        backup_samples = json.load(f)

    all_samples = get_backup_answers(all_samples, {sample['id']: sample for sample in backup_samples})

    precision, recall, f1, support = evaluate_metrics(all_samples)

    return (precision, recall, f1, support)


def full_evaluation(result_file, backup_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)
        
    with open(backup_file, 'r') as f:
        backup_samples = json.load(f)
                
    # all_samples = get_backup_answers(all_samples, {sample['id']: sample for sample in backup_samples})
    
    executable_samples = [sample for sample in all_samples if sample['flag'] == 'success']
    
    # non_executable_samples = [sample for sample in all_samples if sample['flag'] != 'success']
    
    parsing_errors = [sample for sample in all_samples if sample['flag'] == 'parsing error']
    # generation_errors = [sample for sample in all_samples if sample['flag'] == 'generation error']
    execution_errors = [sample for sample in all_samples if sample['flag'] == 'execution error']
    
    print()
    print(f'Executable rate (Exe_Rate): {len(executable_samples)}/{len(all_samples)} ({len(executable_samples)/len(all_samples)})')
    print(f'Parsing errors rate: {len(parsing_errors)}/{len(all_samples)} ({len(parsing_errors)/len(all_samples)})')
    print(f'Execution errors rate: {len(execution_errors)}/{len(all_samples)} ({len(execution_errors)/len(all_samples)})')
    print('-'*75)
    print()
    precision, recall, f1, support = evaluate_metrics(all_samples)
    print("Overall:\n")
    print(f"Average F1: {f1}")
    print(f"Average Precision: {precision}")
    print(f"Average Recall: {recall}")
    print('-'*75)
    print()
    precision, recall, f1, support = evaluate_metrics(executable_samples)
    print("Executables:\n")
    print(f"Average F1: {f1}")
    print(f"Average Precision: {precision}")
    print(f"Average Recall: {recall}")
    print('-'*75)
    print()
    
    precision, recall, f1, support = evaluate_metrics(all_samples, average=None)
    
    for i in range(len(f1)):
        print(f'Choice {index_to_text[i]}')
        print(f'F1: {f1[i]}')
        print(f'Precision: {precision[i]}')
        print(f'Recall: {recall[i]}')
        print('-'*75)
        print()    
    
    return (precision, recall, f1, support)


def get_res(metric):
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    
    res = str(mean) + '^{\pm' + str(std) + '}'
    
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--sketcher_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--refiner_name", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    precisions = []
    f1s = []
    recalls = []
    supports = []
    
    if args.load_dir:
        load_dirs = [args.load_dir]
    else:
        load_dirs = ['outputs_1', 'outputs_2', 'outputs_3']
    
    for load_dir in load_dirs:
        
        if args.refiner_name is None:
            result_path = os.path.join(load_dir, 'logic_inference')
        else:
            result_path = os.path.join(load_dir, 'logic_inference', args.refiner_name)
        
        backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.sketcher_name}.json')
        
        if args.self_refine_round > 0:
            result_file = os.path.join(result_path, f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')
        else:
            result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')

        precision, recall, f1, support = full_evaluation(result_file, backup_file)
        precision, recall, f1, support = precision*100, recall*100, f1*100, support
        # evaluate_predicates(result_file)
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)
        # supports.append(support)
        
    print(f"Precision: {get_res(precisions)}")
    print(f"Recall: {get_res(recalls)}")
    print(f"F1: {get_res(f1s)}")
    # print(f1s)