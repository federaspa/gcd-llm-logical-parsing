import json
import random
import os
from typing import List, Dict, Optional
import copy
import numpy as np

def prepare_samples(experiment_files: List[str], samples_per_file: int, output_path: str):
    """Prepare random samples from multiple experiment files, with each sample duplicated 3 times."""
    samples = []
    for file_path in experiment_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Filter for items containing specific keys
            data = [d for d in data 
                    if 'logic_problem' in d.keys() 
                    or 'logic_problem_gcd' in d.keys() 
                    or 'logic_problem_twosteps' in d.keys()]
            
            # Add source file information
            for item in data:
                item['source_file'] = os.path.basename(file_path)
            
            # Select random samples from the current file
            file_samples = random.sample(data, min(samples_per_file, len(data)))
            
            # Create three independent copies of each sample
            for sample in file_samples:
                # Create three deep copies of the sample
                for _ in range(3):
                    samples.append(copy.deepcopy(sample))
                    
    np.random.shuffle(samples)
    
    # Assign unique annotator_ids to each sample
    for i, sample in enumerate(samples):
        sample['annotator_id'] = i + 1
    
    # Save the results
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    return len(samples)

def main():
            
    samples_per_file = 20
    # Get all JSON files from data directory
    data_dir = 'data'
    experiment_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                    if f.endswith('.json')]
    
    if not experiment_files:
        raise Exception("No JSON files found in the 'data' directory!")
    else:
        prepare_samples(experiment_files, samples_per_file, "prepared_samples.json")
    
if __name__ == "__main__":
    main()