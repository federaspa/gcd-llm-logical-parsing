import numpy as np
import json
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score

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

def evaluate_metrics(QA_results, average="micro"):
    
    predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    return f1_score(gold_answers, predictions, average=average, zero_division=np.nan), precision_score(gold_answers, predictions, average=average, zero_division=np.nan), recall_score(gold_answers, predictions, average=average, zero_division=np.nan)

def get_backup_answers(samples, backup):
    
    non_executable_samples = [sample for sample in samples if sample['flag'] != 'success']
    
    for sample in non_executable_samples:
        sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
        # sample['predicted_reasoning'] = backup[sample['id']]['predicted_reasoning']
        
    # print(f"Recovered {len(non_executable_samples)} samples from backup")
        
    return samples


def full_evaluation(result_file, backup_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)
        
    # with open(backup_file, 'r') as f:
    #     backup_samples = json.load(f)

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
    f1, precision, recall = evaluate_metrics(all_samples)
    print(f"Average F1: {f1}")
    print(f"Average Precision: {precision}")
    print(f"Average Recall: {recall}")
    print('-'*75)
    print()
    
    # f1, precision, recall = evaluate_metrics(all_samples, average=None)
    
    # for i in range(len(f1)):
    #     print(f'Choice {index_to_text[i]}')
    #     print(f'F1: {f1[i]}')
    #     print(f'Precision: {precision[i]}')
    #     print(f'Recall: {recall[i]}')
    #     print('-'*75)
    #     print()    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--sketcher_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='dynamic')
    parser.add_argument("--result_path", type=str, default='./outputs/logic_inference')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    
    backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.sketcher_name}.json')
    
    if args.self_refine_round > 0:
        result_file = os.path.join(result_path, f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')
    else:
        result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{args.sketcher_name}_{args.prompt_mode}.json')

    # evaluate_predicates(result_file)
    full_evaluation(result_file, backup_file)