import unittest
import json
import os
import yaml
from pathlib import Path
from typing import Set, List
import re

def get_models(models_path: str) -> list:
    """
    Discover available models by examining the model directory structure.
    
    Args:
        models_path: Path to the directory containing model folders
        model_name_filter: Optional filter string to limit which models are included by name
        max_model_size: Optional maximum model size filter (e.g., "7b", "14b", "32b")
        
    Returns:
        List of model names sorted by model size (largest first)
    """
    models = []
    models_path = Path(models_path)
    
    # Scan through all subdirectories in the models path
    for model_dir in [d for d in models_path.iterdir() if d.is_dir()]:
        base_name = model_dir.name
        
        # Look for single .gguf files (excluding multi-part files)
        single_gguf_files = []
        for file_path in model_dir.glob("*.gguf"):
            # Skip files that match the multi-part pattern
            if not re.search(r'-\d+-of-\d+\.gguf$', str(file_path)):
                single_gguf_files.append(file_path)
        
        # Look for multi-part model files
        multipart_models = set()
        for pattern in model_dir.glob("*-*-of-*.gguf"):
            # Extract the model name without the part number
            # Example: mistral-7b-v0.1-0-of-3.gguf -> mistral-7b-v0.1
            part_match = re.match(r'(.*)-\d+-of-\d+\.gguf', pattern.name)
            if part_match:
                model_name = part_match.group(1)
                multipart_models.add(model_name)
        
        # Add all found models, applying filters
        for model_file in single_gguf_files:
            model_name = model_file.stem
                    
            models.append(model_name)
                
        # Add multi-part models
        for model_name in multipart_models:
                    
            models.append(model_name)
    
    # Sort models by size (largest first)
    return models

class TestOutputFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        # Default paths - can be overridden with environment variables
        cls.save_path = os.getenv('SAVE_PATH', './outputs')
        cls.models_path = os.getenv('MODELS_PATH', '/home/fraspanti/LLMs')
        cls.data_path = os.getenv('DATA_PATH', './data')
        cls.dataset_name = os.getenv('DATASET_NAME', 'GSM8K_symbolic')
        cls.shots = os.getenv('SHOTS_NUMBER', '5shots')
        
        # Load model names from configs
        cls.model_names = get_models(cls.models_path)
        print(f"\nFound {len(cls.model_names)} model configurations")
        
        # Load sampled IDs
        sampled_ids_path = Path(cls.data_path) / cls.dataset_name / 'sampled_ids.json'
        with open(sampled_ids_path) as f:
            cls.sampled_ids = set(json.load(f))
        print(f"Loaded {len(cls.sampled_ids)} sampled IDs")
        
        # Find output files for each model
        cls.output_files = []
        save_path = Path(cls.save_path)
        shots_dir = save_path / cls.shots
        if shots_dir.is_dir():
            logic_problems_dir = shots_dir / "logic_problems"
            if logic_problems_dir.exists():
                for json_file in logic_problems_dir.glob("*.json"):
                    # Check if file matches dataset name and any model name
                    if cls.dataset_name in json_file.stem and any(model in json_file.stem for model in cls.model_names):
                        cls.output_files.append(json_file)
        
        print(f"Found {len(cls.output_files)} output files matching dataset '{cls.dataset_name}' in {cls.shots}")
        for file in cls.output_files:
            print(f"  - {file.name}")

    def test_1_json_validity(self):
        """Test if all output files are valid JSON"""
        print(f"\nTesting JSON validity")
        for file_path in self.output_files:
            with self.subTest(file=file_path):
                try:
                    with open(file_path) as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    self.fail(f"Invalid JSON in {file_path}: {str(e)}")

    def test_2_sampled_ids_presence(self):
        """Test if all sampled IDs are present in each output file"""
        print(f"\nTesting IDs presence")
        for file_path in self.output_files:
            with self.subTest(file=file_path):
                with open(file_path) as f:
                    data = json.load(f)
                    output_ids = {sample['id'] for sample in data}
                    
                    # Check for missing IDs
                    missing_ids = self.sampled_ids - output_ids
                    if missing_ids:
                        self.fail(f"{file_path} is missing {len(missing_ids)} IDs")
                    
                    # Check for extra IDs
                    extra_ids = output_ids - self.sampled_ids
                    if extra_ids:
                        self.fail(f"{file_path} has {len(extra_ids)} extra IDs")

    def test_3_all_models_present(self):
        """Test if there's an output file for each model configuration"""
        print(f"\nTesting model coverage")
        
        # Find which models have output files
        models_with_files = set()
        for file_path in self.output_files:
            for model_name in self.model_names:
                if model_name in file_path.stem:
                    models_with_files.add(model_name)
                    break
        
        # Check for missing models
        missing_models = set(self.model_names) - models_with_files
        if missing_models:
            self.fail(f"Missing output files for models: {', '.join(sorted(missing_models))}")

if __name__ == '__main__':
    unittest.main(verbosity=0)