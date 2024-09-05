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

def load_outputs(dataset_name, sketcher1, sketcher2):
    
    with open(f'outputs/outputs_1/logic_inference/no_gcd/llama-2-7b/self-refine-3_{dataset_name}_dev_{sketcher1}_dynamic.json', 'r') as f:
        outs3 = json.load(f)

    with open(f'outputs/outputs_1/logic_inference/no_gcd/llama-2-7b/self-refine-3_{dataset_name}_dev_{sketcher2}_dynamic.json', 'r') as f:
        outs4 = json.load(f)
        
    return outs3, outs4
        
def average_premises(sample):   
    context = sample['context_fol']
    return len(context)
        
def group_by_premises(ds, outs3, outs4):
    
    grouped = {}
    
    for (out3, out4) in zip(outs3, outs4):
        
        assert out3['id'] == out4['id'], 'Different ids in raw and gcd outputs'
        
        if out3['id'] == 1:
            print(out3['predicted_answer'])
            print(out4['predicted_answer'])
        
        sample = [s for s in ds if s['id'] == out3['id']]
        
        if len(sample) != 1:
            print(f'No or too many samples with id: {out3["id"]}')
            continue
        else:
            sample = sample[0]
        
        av_premises = average_premises(sample)
        
        point = {
            'id': sample['id'],
            'predicted_answer_3': out3['predicted_answer'],
            'predicted_answer_4': out4['predicted_answer'],
            'answer': sample['answer'],
        }
        
        if not av_premises in grouped.keys():
            grouped[av_premises] = []
        grouped[av_premises].append(point)
        
    return grouped

def main():
    
    plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(18, 10))

        
    folio, nli = load_datasets()
    
    ds = folio
    name = 'FOLIO'
    
    

    outs3, outs4 = load_outputs(name, 'gpt-3.5-turbo', 'gpt-4o')
    
    assert len(outs3) == len(outs4), 'Different number of samples in raw and gcd outputs'
    
    grouped = group_by_premises(ds, outs3, outs4)
    
    f1s = {}
        
    for key, value in grouped.items():
        
        y_true = [sample['answer'] for sample in value]
        y_pred_raw = [sample['predicted_answer_3'] for sample in value]
        y_pred_gcd = [sample['predicted_answer_4'] for sample in value]
        
        f1s[key] = {'raw': f1_score(y_true, y_pred_raw, average="weighted"), 'gcd': f1_score(y_true, y_pred_gcd, average="weighted")}
        
    # Order by key
    f1s = OrderedDict(sorted(f1s.items())) 
    
    # Plot F1 scores with seaborn
    sns.lineplot(x=list(f1s.keys()), y=[f1['raw'] for f1 in f1s.values()], ax=ax, label='GPT-3.5-Turbo')
    sns.lineplot(x=list(f1s.keys()), y=[f1['gcd'] for f1 in f1s.values()], ax=ax, label='GPT-4')
    ax.set_ylabel('Weighted F1 score')
    ax.set_title(f'Weighted F1 score by number of premises')
    ax.legend()
        
        
        # plt.plot(list(f1s.keys()), [f1['raw'] for f1 in f1s.values()], label='Raw')
        # plt.plot(list(f1s.keys()), [f1['gcd'] for f1 in f1s.values()], label='GCD')
        # plt.xlabel('Premises per sample')
        # plt.ylabel('F1 score')
        # plt.title(f'{name} - F1 score by premises per sample')
        # plt.legend()
    ax.set_xlabel('Number of premises')
    plt.savefig(f'evaluation/qualitative/images/f1_by_premises.png')
    
    
    # print('--------------------------------------------------')

        

if __name__ == "__main__":
    main()