import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_datasets():
    with open('data/FOLIO/dev.json', 'r') as f:
        dev_folio = json.load(f)
        
    with open('data/LogicNLI/dev.json', 'r') as f:
        dev_nli = json.load(f)
    
    with open('data/FOLIO/train.json', 'r') as f:
        train_folio = json.load(f)
        
    with open('data/LogicNLI/train.json', 'r') as f:
        train_nli = json.load(f)
        
    folio = train_folio + dev_folio
    nli = train_nli + dev_nli
    
    return folio, nli


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

def average_premises(ds):   
    all_counts  = []
    
    for i, sample in enumerate(ds):
        context = sample['context_fol']
        all_counts.append(len(context))
        
    mean = np.mean(all_counts)
        
    return all_counts, mean

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

def main():
    
    folio, nli = load_datasets()
    
    for ds, name in zip([folio, nli], ['FOLIO', 'LogicNLI']):
        counts, mean = words_per_premise(ds)
        
        print(f'{name} - Mean words per premise: {mean}')
        
        fig, _ = plt.subplots()
        # plot counts
        sns.histplot(counts)
        plt.title(f'Words per premise - {name}')
        plt.savefig(f'evaluation/qualitative/images/words_per_premise_{name}.png')
        
        
        fig, _ = plt.subplots()
        
        counts, mean = nestings_per_premise(ds)
        
        print(f'{name} - Mean nesting per premise: {mean}')
        
        # plot counts
        sns.histplot(counts)  
        plt.title(f'Nesting per premise - {name}')
        plt.savefig(f'evaluation/qualitative/images/nesting_per_premise{name}.png')
        
        
        fig, _ = plt.subplots()
        
        counts, mean = predicates_per_premise(ds)
        
        print(f'{name} - Mean predicates per premise: {mean}')
        
        # plot counts
        sns.histplot(counts)
        plt.title(f'Predicates per premise - {name}')
        plt.savefig(f'evaluation/qualitative/images/predicates_per_premise{name}.png')
        
        print(f'{name} - Mean premises per sample: {mean}')
        
        
        print('--------------------------------------------------')
        
        counts, mean = average_premises(ds)
        
        

if __name__ == "__main__":
    main()