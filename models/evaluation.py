import re
import json
from tqdm import tqdm
import random
import os
import argparse
import numpy as np
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

def compute_accuracy(samples):
    
    gold_answers, predictions = parse_answers(samples)
    
    if len(gold_answers) != len(predictions):
        raise ValueError("The lists of gold answers and predictions must have the same length.")
    
    correct_predictions = sum(gold == pred for gold, pred in zip(gold_answers, predictions))
    total_predictions = len(gold_answers)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return accuracy
        

def evaluate_metrics(QA_results, average='weighted'):
    
    predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    # return f1_score(gold_answers, predictions, average=average, zero_division=np.nan), precision_score(gold_answers, predictions, average=average, zero_division=np.nan), recall_score(gold_answers, predictions, average=average, zero_division=np.nan)
    
    return precision_recall_fscore_support(gold_answers, predictions, average=average, zero_division=np.nan)

def parse_answers(samples):
    
    def get_choice(answer_str):
        choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
                'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.', 'N/A']
        for c in choices:
            if answer_str.startswith(c):
                return c.replace(')', '')
        return None



    gold_answers = []
    predictions = []

    for sample in samples:
        gold_answer = sample['answer']
        answer_pred = sample['predicted_answer'].strip()
        prediction = get_choice(answer_pred)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is']
        if prediction is None:
            print(answer_pred)
            for indicator in indicators:
                if answer_pred.find(indicator)>=0:
                    answer_pred = answer_pred.split(indicator)[1].strip()
                    prediction = get_choice(answer_pred)
                    break
                
        gold_answers.append(gold_answer)
        predictions.append(prediction)
        
    return gold_answers, predictions


def get_backup_answers(samples, backup):
    
    non_executable_samples = [sample for sample in samples if sample['flag'] != 'success']
    
    for sample in non_executable_samples:
        sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
        
    return samples

def full_evaluation(result_file):
    
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    executable_samples = [sample for sample in all_samples if sample['status'] == 'success']

    print(compute_accuracy(all_samples))
    print(compute_accuracy(executable_samples))



def get_res(metric):
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    
    res = str(mean) + '^{\pm' + str(std) + '}'
    
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--sketcher_path', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./outputs/logic_inference')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    sketcher_name = os.path.splitext(args.sketcher_path)[0].split('/')[-1]
    
    result_file = os.path.join(args.result_path, f'{args.dataset_name}_{args.split}_{sketcher_name}.json')
    
    # backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.model_name}.json')
    
    full_evaluation(result_file)