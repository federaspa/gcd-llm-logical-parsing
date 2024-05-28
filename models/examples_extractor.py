import os
import json
import torch
import openai
import argparse

import numpy as np

from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import util


load_dotenv()  # take environment variables from .env.
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

class ExampleExtractionEngine:
    def __init__(self, args):
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.source_split = args.source_split
        self.target_split = args.target_split
        self.max_examples = args.max_examples
        
        

        self.source_dataset, self.target_dataset = self.load_data()
        
        self.save_path = os.path.join(self.data_path, self.dataset_name, f'{self.target_split}_examples.json')
        
    def load_data(self):
        source_path = os.path.join(self.data_path, self.dataset_name, f'{self.source_split}.json')
        target_path = os.path.join(self.data_path, self.dataset_name, f'{self.target_split}.json')
        
        with open(source_path, 'r') as f:
            source_dataset = json.load(f)
        with open(target_path, 'r') as f:
            target_dataset = json.load(f)
            
        return source_dataset, target_dataset


    # def make_sentences(self):
        

    def extract_examples(self):
        
        sorted_stories = {}
            
        max_examples = min(self.max_examples, len(self.source_dataset))
        
        source_sentences = []
        
        for source_story in self.source_dataset:
            
            seen_source_stories = set()
            
            if source_story['story_id'] in seen_source_stories:
                continue
            
            if 'question_fol' not in source_story.keys():
                continue
            
            source_sentence = ' '.join(source_story['context'])
            source_sentence += f" {source_story['question']}"
            
            source_sentences.append(source_sentence)
            
            seen_source_stories.add(source_story['story_id'])
        
        for target_story in tqdm(self.target_dataset):
            
            target_sentence = ' '.join(target_story['context'])
            target_sentence += f" {target_story['question']}"
            
            response_source = openai.Embedding.create(
                input=source_sentences,
                model="text-embedding-3-small"
            )

            response_target = openai.Embedding.create(
                input=target_sentence,
                model="text-embedding-3-small"
            )

            embeddings1 = torch.tensor([d.embedding for d in response_source.data])
            embeddings2 = torch.tensor(response_target.data[0].embedding)

            cosine_scores = util.cos_sim(embeddings1, embeddings2)
                    
            if cosine_scores.device == 'cpu':
                cosine_scores = cosine_scores.numpy()
            else:
                cosine_scores = cosine_scores.cpu().numpy()
                
            sorted_indexes = np.argsort(cosine_scores, axis=0)[::-1]
            
            
            sorted_stories[target_story['id']] = [self.source_dataset[i[0]] for i in sorted_indexes[:max_examples]]
        
        with open(self.save_path, 'w') as f:
            json.dump(sorted_stories, f, indent=2, ensure_ascii=False)
            
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--source_split', type=str, default='train')
    parser.add_argument('--target_split', type=str, default='dev')
    parser.add_argument('--max_examples', type=int, default=3)
    # parser.add_argument('--save_path', type=str, default='./outputs/dynamic_examples')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = ExampleExtractionEngine(args)
    engine.extract_examples()