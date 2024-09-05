import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import f1_score
from collections import OrderedDict


with open('evaluation/quantitative/wrong_ids.json', 'r') as f:
    wrong_ids = json.load(f)

def load_datasets():
    with open('data/FOLIO/dev.json', 'r') as f:
        folio = json.load(f)
        
    with open('data/LogicNLI/dev.json', 'r') as f:
        nli = json.load(f)
        
    nli = [sample for sample in nli if sample['id'] not in wrong_ids]
        
    return folio, nli

def load_outputs(dataset_name, sketcher):
    
    with open(f'outputs/outputs_1/logic_inference/no_gcd/llama-2-7b/self-refine-3_{dataset_name}_dev_{sketcher}_dynamic.json', 'r') as f:
        raw_outs = json.load(f)

    with open(f'outputs/outputs_1/logic_inference/gcd/llama-2-7b/self-refine-3_{dataset_name}_dev_{sketcher}_dynamic.json', 'r') as f:
        gcd_outs = json.load(f)
        
    if dataset_name == 'LogicNLI':
        raw_outs = [sample for sample in raw_outs if sample['id'] not in wrong_ids]
        gcd_outs = [sample for sample in gcd_outs if sample['id'] not in wrong_ids]
        
    return raw_outs, gcd_outs
        
def words_per_premise(ds):
    
    all_counts  = []
    
    for sample in ds:
        context = sample['context']
        sample_counts = []
        for sentence in context:
            # remove punctuation
            sentence = sentence.replace('.', '')
            sentence = sentence.replace(',', '')
            sentence = sentence.replace('?', '')
            sentence = sentence.replace('!', '')
            count = len(set(sentence.split()))
            
            sample_counts.append(count)
            
        all_counts.append(np.mean(sample_counts))
        
    mean = np.mean(all_counts)
        
    return all_counts, mean

def nestings_per_premise(ds):
    
    all_counts  = []
    
    for i, sample in enumerate(ds):
        context = sample['context_fol']
        sample_counts = []
        for sentence in context:
            
            if sentence.startswith('('):
                sentence = sentence[1:]
            
            predicates = re.findall(r'[A-Z]\w+\([a-z|,|\s]*\)', sentence)
            
            for predicate in predicates:
                sentence = sentence.replace(predicate, '')
                
            # find max number of open brackets in a row
            max_count = 0
            count = 0
            for char in sentence:
                if char == '(':
                    count += 1
                    if count > max_count:
                        max_count = count
                else:
                    count = 0
                        
            if max_count == 0:
                continue
            
            sample_counts.append(max_count)
        
        if len(sample_counts) == 0:
            continue    
        
        all_counts.append(np.mean(sample_counts))
        
    mean = np.mean(all_counts)
        
    return all_counts, mean

def average_premises(sample):   
    context = sample['context_fol']
    return len(context)
        
def predicates_per_premise(ds):
    
    all_counts  = []
    
    for i, sample in enumerate(ds):
        context = sample['context_fol']
        sample_counts = []
        for sentence in context:
            
            
            predicates = re.findall(r'[A-Z]\w+\([a-z|,|\s]*\)', sentence)
            
            if len(predicates) == 0:
                continue
            
            sample_counts.append(len(predicates))
        
        if len(sample_counts) == 0:
            continue  
        
        all_counts.append(np.mean(sample_counts))
        
    mean = np.mean(all_counts)
        
    return all_counts, mean

def group_by_premises(ds, raw_outs, gcd_outs):
    
    grouped = {}
    
    for (raw_out, gcd_out) in zip(raw_outs, gcd_outs):
        
        assert raw_out['id'] == gcd_out['id'], 'Different ids in raw and gcd outputs'
        
        if raw_out['id'] == 1:
            print(raw_out['predicted_answer'])
            print(gcd_out['predicted_answer'])
        
        sample = [s for s in ds if s['id'] == raw_out['id']]
        
        if len(sample) != 1:
            print(f'No or too many samples with id: {raw_out["id"]}')
            continue
        else:
            sample = sample[0]
        
        av_premises = average_premises(sample)
        
        point = {
            'id': sample['id'],
            'predicted_answer_raw': raw_out['predicted_answer'],
            'predicted_answer_gcd': gcd_out['predicted_answer'],
            'answer': sample['answer'],
        }
        
        if not av_premises in grouped.keys():
            grouped[av_premises] = []
        grouped[av_premises].append(point)
        
    return grouped

def main():
    
    plt.rcParams.update({'font.size': 26, 'font.weight': 'bold'})
    sns.set_style("whitegrid")
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20, 20))

        
    folio, nli = load_datasets()
    
    ds = folio
    name = 'FOLIO'
    
    
    for sketcher, ax in zip(['gpt-3.5-turbo', 'gpt-4o'], [ax1, ax2]):
    
        raw_outs, gcd_outs = load_outputs(name, sketcher)
        
        assert len(raw_outs) == len(gcd_outs), 'Different number of samples in raw and gcd outputs'
        
        grouped = group_by_premises(ds, raw_outs, gcd_outs)
        
        f1s = {}
            
        for key, value in grouped.items():
            
            y_true = [sample['answer'] for sample in value]
            y_pred_raw = [sample['predicted_answer_raw'] for sample in value]
            y_pred_gcd = [sample['predicted_answer_gcd'] for sample in value]
            
            f1s[key] = {'raw': f1_score(y_true, y_pred_raw, average="weighted"), 'gcd': f1_score(y_true, y_pred_gcd, average="weighted")}
            
        # Order by key
        f1s = OrderedDict(sorted(f1s.items())) 
        
        # Plot F1 scores with seaborn
        sns.lineplot(x=list(f1s.keys()), y=[f1['raw'] for f1 in f1s.values()], ax=ax, label='No GCD')
        sns.lineplot(x=list(f1s.keys()), y=[f1['gcd'] for f1 in f1s.values()], ax=ax, label='GCD')
        ax.set_ylabel('Weighted F1 score')
        ax.set_title(f'Weighted F1 score by number of premises - {'GPT-3.5-Turbo' if sketcher == 'gpt-3.5-turbo' else 'GPT-4o'}')
        ax.legend()
        
        
        # plt.plot(list(f1s.keys()), [f1['raw'] for f1 in f1s.values()], label='Raw')
        # plt.plot(list(f1s.keys()), [f1['gcd'] for f1 in f1s.values()], label='GCD')
        # plt.xlabel('Premises per sample')
        # plt.ylabel('F1 score')
        # plt.title(f'{name} - F1 score by premises per sample')
        # plt.legend()
    ax2.set_xlabel('Number of premises')
    plt.savefig(f'evaluation/qualitative/images/f1_by_premises.png')
    
    
    # print('--------------------------------------------------')

        

if __name__ == "__main__":
    main()