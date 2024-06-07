import re
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
    'H': 7
}

index_to_text = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H'
}
    
# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '')

    if answer_str.startswith(':'):
       return answer_str.replace(':', '').replace('.', '').strip()
    return None


def evaluate_metrics(QA_results, average="micro"):
    
    predictions = [text_to_index[sample['predicted_answer']] for sample in QA_results]
    gold_answers = [text_to_index[sample['answer']] for sample in QA_results]
    
    return f1_score(gold_answers, predictions, average=average), precision_score(gold_answers, predictions, average=average), recall_score(gold_answers, predictions, average=average)

def full_evaluation(result_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    executable_samples = [sample for sample in all_samples if sample['flag'] == 'success']
    non_executable_samples = [sample for sample in all_samples if sample['flag'] != 'success']
    
    parsing_errors = [sample for sample in all_samples if sample['flag'] == 'parsing error']
    generation_errors = [sample for sample in all_samples if sample['flag'] == 'generation error']
    execution_errors = [sample for sample in all_samples if sample['flag'] == 'execution error']
    
    print()
    print(f'Executable rate (Exe_Rate): {len(executable_samples)}/{len(all_samples)} ({len(executable_samples)/len(all_samples)})')
    print(f'Parsing errors rate: {len(parsing_errors)}/{len(all_samples)} ({len(parsing_errors)/len(all_samples)})')
    print(f'Execution errors rate: {len(execution_errors)}/{len(all_samples)} ({len(execution_errors)/len(all_samples)})')
    print('-'*75)
    print()
    f1, precision, recall = evaluate_metrics(executable_samples)
    print(f"Weighted F1: {f1}")
    print(f"Weighted Precision: {precision}")
    print(f"Weighted Recall: {recall}")
    print('-'*75)
    print()
    
    f1, precision, recall = evaluate_metrics(executable_samples, average=None)
    
    for i in range(len(f1)):
        print(f'Choice {index_to_text[i]}')
        print(f'F1: {f1[i]}')
        print(f'Precision: {precision[i]}')
        print(f'Recall: {recall[i]}')
        print('-'*75)
        print()
    
    # print(f'F1 scores for executable samples: {f1}')
    # print(f'Precision scores for executable samples: {precision}')
    # print(f'Recall scores for executable samples: {recall}')
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument('--self_refine_round', type=int, default=0)
    parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument('--prompt_mode', type=str, choices=['dynamic', 'static'], default='static')
    parser.add_argument('--backup', type=str, default='random', choices=['random', 'Direct', 'CoT'])
    parser.add_argument("--result_path", type=str, default='./outputs/logic_inference')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_path
    if args.self_refine_round > 0:
        result_file = os.path.join(result_path, f'self-refine-{args.self_refine_round}_{args.dataset_name}_{args.split}_{args.model_name}_{args.prompt_mode}_backup-{args.backup}.json')
    else:
        result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{args.model_name}_{args.prompt_mode}_backup-{args.backup}.json')

    full_evaluation(result_file)