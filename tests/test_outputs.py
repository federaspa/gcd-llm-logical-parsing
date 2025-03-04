import unittest
import json
import os
import yaml
from pathlib import Path
from typing import Set, List

def get_model_names() -> List[str]:
    """Get base model names from config files."""
    config_path = Path('configs/models')
    model_names = []
    
    for config_file in config_path.glob('*.yml'):
        base_name = config_file.stem
        if base_name not in model_names:
            model_names.append(base_name)
            
    return model_names

class TestOutputFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods."""
        # Default paths - can be overridden with environment variables
        cls.save_path = os.getenv('SAVE_PATH', './outputs')
        cls.data_path = os.getenv('DATA_PATH', './data')
        cls.dataset_name = os.getenv('DATASET_NAME', 'GSM8K_symbolic')
        cls.shots = os.getenv('SHOTS_NUMBER', '5shots')
        
        # Load model names from configs
        cls.model_names = get_model_names()
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