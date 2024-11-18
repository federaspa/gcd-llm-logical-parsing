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
        
def compute_metrics(all_samples):
    
    executable_samples = [sample for sample in all_samples if sample['status'] != 'parsing error']
        
    total_accuracy = compute_accuracy(all_samples)
    covered_accuracy = compute_accuracy(executable_samples)
    coverage = len(executable_samples)/len(all_samples) if len(all_samples) else 0
    
    return total_accuracy, covered_accuracy, coverage
    
    
# def evaluate_metrics(QA_results, average='weighted'):
    
#     predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
#     gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
#     # return f1_score(gold_answers, predictions, average=average, zero_division=np.nan), precision_score(gold_answers, predictions, average=average, zero_division=np.nan), recall_score(gold_answers, predictions, average=average, zero_division=np.nan)
    
#     return precision_recall_fscore_support(gold_answers, predictions, average=average, zero_division=np.nan)

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
    
    non_executable_samples = [sample for sample in samples if sample['status'] != 'success']
    
    for sample in non_executable_samples:
        sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
        
    return samples

def both_success_results(samples):
    success_samples = []
    
    for s in samples:
        unc = s.get('logic_problem', {})
        con = s.get('logic_problem_gcd', {})
        
        unc_flag = unc.get('status', '')
        con_flag = con.get('status', '')
        
        
        if unc_flag == con_flag == 'success':
            success_samples.append(s)
            
    unconstrained = [s.get('logic_problem', {}) for s in success_samples]
    unconstrained = [s for s in unconstrained if s]
    
    constrained = [s.get('logic_problem_gcd', {}) for s in success_samples]
    constrained = [s for s in constrained if s]
    
    unconstrained_metrics = compute_metrics(unconstrained)
    constrained_metrics = compute_metrics(constrained)
            
    return unconstrained_metrics, constrained_metrics

def constrained_success_only_results(samples):
    con_success_samples = []
    
    for s in samples:
        unc = s.get('logic_problem', {})
        con = s.get('logic_problem_gcd', {})
        
        unc_flag = unc.get('status', '')
        con_flag = con.get('status', '')
        
        if (unc_flag != con_flag) and (con_flag == 'success'):
            con_success_samples.append(s)
    
    constrained = [s.get('logic_problem_gcd', {}) for s in con_success_samples]
    constrained = [s for s in constrained if s]
    
    constrained_metrics = compute_metrics(constrained)
            
    return constrained_metrics

def full_evaluation(result_file):
    
    results = {'ALL': {}, 'BOTH_SUCCESS': {}}
    
    with open(result_file, 'r') as f:
        raw_samples = json.load(f)
        
    unconstrained = [s.get('logic_problem', {}) for s in raw_samples]
    unconstrained = [s for s in unconstrained if s]
    
    constrained = [s.get('logic_problem_gcd', {}) for s in raw_samples]
    constrained = [s for s in constrained if s]
        
    results['ALL']['UNCONSTRAINED'] = compute_metrics(unconstrained)
    results['ALL']['CONSTRAINED'] = compute_metrics(constrained)

    results['BOTH_SUCCESS']['UNCONSTRAINED'], results['BOTH_SUCCESS']['CONSTRAINED'] = both_success_results(raw_samples)
    results['CONSTRAINED_SUCCESS'] = constrained_success_only_results(raw_samples)

    print('Evaluating file\n', result_file)

    for cat in ['ALL', 'BOTH_SUCCESS']:
        print(F'{cat}\n')
        
        for subcat in ['UNCONSTRAINED', 'CONSTRAINED']:
            print(F'{subcat}\n')
            
            for name, value in zip(['Total accuracy', 'Covered accuracy', 'Coverage'], results[cat][subcat]):
                print(f'{name}: {value:.2}')
                print()
                
    print('CONSTRAINED_SUCCESS\n')
    for name, value in zip(['Total accuracy', 'Covered accuracy', 'Coverage'], results['CONSTRAINED_SUCCESS']):
        print(f'{name}: {value:.2}')
        print()
                
    



def get_res(metric):
    mean = round(np.mean(metric), 2)
    std = round(np.std(metric), 1)
    
    res = str(mean) + '^{\pm' + str(std) + '}'
    
    return res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--sketcher-name', type=str, required=True)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--self-refine-round', type=int, default=0)
    parser.add_argument('--result-path', type=str, default='./outputs/logic_inference')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
      
    if args.self_refine_round > 0:
        programs_file = f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.sketcher_name}.json'
    else:
        programs_file = f'{args.dataset_name}_{args.split}_{args.sketcher_name}.json'
    
    result_file = os.path.join(args.result_path, programs_file)
    
    
    # backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.model_name}.json')
    
    full_evaluation(result_file)