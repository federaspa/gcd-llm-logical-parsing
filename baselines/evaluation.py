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

# def extract_number(string):
#     # Remove all characters except digits, decimal point and negative sign
#     try:
#         num_string = re.sub(r'[^\d.-]', '', string)
#         num_string = num_string.replace('$', '')
#         return float(num_string)
#     except:
#         try:
#             return float(random.randint(0, 100))
#             # return float(w2n.word_to_num(string))
#         except:
#             # print('Error: ', string)
#             print('Error')
#             return float(random.randint(0, 100))

# def argmax(iterable):
#     return max(enumerate(iterable), key=lambda x: x[1])[0]

# these functions are heavily influenced by the HF squad_metrics.py script
# def normalize_text(s):
#     """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
#     import string, re

#     def remove_articles(text):
#         regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#         return re.sub(regex, " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def compute_exact_match(prediction, truth):
#     return int(normalize_text(prediction) == normalize_text(truth))
#     # return prediction == truth

# def compute_f1(prediction, truth):
#     pred_tokens = normalize_text(prediction).split()
#     truth_tokens = normalize_text(truth).split()
    
#     # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
#     if len(pred_tokens) == 0 or len(truth_tokens) == 0:
#         return int(pred_tokens == truth_tokens)
    
#     common_tokens = set(pred_tokens) & set(truth_tokens)
    
#     # if there are no common tokens then f1 = 0
#     if len(common_tokens) == 0:
#         return 0
    
#     prec = len(common_tokens) / len(pred_tokens)
#     rec = len(common_tokens) / len(truth_tokens)
    
#     return 2 * (prec * rec) / (prec + rec)

# def evaluate_sample(prediction, gold_answers):
#     em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
#     f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
#     return em_score, f1_score

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')
    return 'N/A'

def evaluate_metrics(QA_results, average='weighted'):
    
    predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    # return f1_score(gold_answers, predictions, average=average, zero_division=np.nan), precision_score(gold_answers, predictions, average=average, zero_division=np.nan), recall_score(gold_answers, predictions, average=average, zero_division=np.nan)
    
    return precision_recall_fscore_support(gold_answers, predictions, average=average, zero_division=np.nan)

def parse_answers(result_file):
    with open(result_file, 'r') as f:
        QA_results = json.load(f)

    for sample in QA_results:
        # gold_answer = sample['answer'].replace('(', '').replace(')', '').strip()
        answer_str = sample['predicted_answer'].strip()
        prediction = get_choice(answer_str)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is']
        if prediction is None:
            for indicator in indicators:
                if answer_str.find(indicator)>=0:
                    answer_str = answer_str.split(indicator)[1].strip()
                    prediction = get_choice(answer_str)
                    break
        
        sample['predicted_answer'] = prediction

        # if prediction is None:
        #     sample['predicted_answer'] = 'N/A'
        #     non_executable_samples.append(sample)
        #     continue
        # else:
        #     executable_samples.append(sample)
    
    return QA_results


def get_backup_answers(samples, backup):
    
    non_executable_samples = [sample for sample in samples if sample['flag'] != 'success']
    
    for sample in non_executable_samples:
        sample['predicted_answer'] = backup[sample['id']]['predicted_answer']
        # sample['predicted_reasoning'] = backup[sample['id']]['predicted_reasoning']
        
    # print(f"Recovered {len(non_executable_samples)} samples from backup")
        
    return samples

def partial_evaluation(result_file):
    all_samples = parse_answers(result_file)

    precision, recall, f1, support = evaluate_metrics(all_samples)

    return (precision, recall, f1, support)

def full_evaluation(result_file):

    all_samples = parse_answers(result_file)
    
    # with open(backup_file, 'r') as f:
    #     backup_samples = json.load(f)
    
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
    precision, recall, f1, support = evaluate_metrics(all_samples)
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
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./results')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    result_file = os.path.join(args.result_path, f'{args.mode}_{args.dataset_name}_{args.split}_{args.model_name}.json')
    
    backup_file = os.path.join('./baselines/results', f'CoT_{args.dataset_name}_{args.split}_{args.model_name}.json')
    
    precision, recall, f1, support = partial_evaluation(result_file)
    precision, recall, f1, support = precision*100, recall*100, f1*100, support
    # evaluate_predicates(result_file)
    
    print(f"Precision: {get_res([precision])}")
    print(f"Recall: {get_res([recall])}")
    print(f"F1: {get_res([f1])}")