import os
import json
import argparse
import traceback
from typing import Dict, List, Any
from tqdm import tqdm
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    sketcher_path: str
    refiner_path: str
    dataset_name: str
    self_refine_round: int
    data_path: str
    save_path: str
    split: str
    gcd: bool

class FileHandler:
    @staticmethod
    def load_json(file_path: Path) -> Any:
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {file_path}")
            return None
        except IOError:
            print(f"Error reading file {file_path}")
            return None

    @staticmethod
    def save_json(data: Any, file_path: Path) -> None:
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            print(f"Error writing to file {file_path}")

class ErrorAnnotator:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = self.load_logic_problems()
        self.error_types_global: Dict[str, Dict[str, int]] = {}

    def load_logic_problems(self) -> List[Dict[str, Any]]:
        prefix = f'self-refine-{self.config.self_refine_round}_' if self.config.self_refine_round > 0 else ""
        
        if self.config.self_refine_round > 0:
            programs_path = Path(self.config.save_path).parent / 'refinement' / ('GCD' if self.config.gcd else 'NO_GCD') / self.get_refiner_name() / f'{prefix}{self.config.dataset_name}_{self.config.split}_{self.get_sketcher_name()}.json'
        else:
            programs_path = Path(self.config.save_path).parent / 'logic_problems' / f'{self.config.dataset_name}_{self.config.split}_{self.get_sketcher_name()}.json'
        
        dataset = FileHandler.load_json(programs_path)
        if dataset:
            print(f"Loaded {len(dataset)} examples from {self.config.split} split.")
        return dataset or []

    def get_sketcher_name(self) -> str:
        return Path(self.config.sketcher_path).stem

    def get_refiner_name(self) -> str:
        return Path(self.config.refiner_path).stem

    def run(self) -> None:
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        prefix = f'self-refine-{self.config.self_refine_round}_' if self.config.self_refine_round > 0 else ""
        analysis_file = Path("./analysis_results") / f"{prefix}{self.config.dataset_name}_{self.config.split}_{self.get_sketcher_name()}_error_types.json"

        self.error_types_global = FileHandler.load_json(analysis_file) or {}
        
        save_file = save_path / f'{prefix}{self.config.dataset_name}_{self.config.split}_{self.get_sketcher_name()}.json'
        
        error_types_round: Dict[str, int] = {}
        outputs: List[Dict[str, Any]] = []

        for sample in self.dataset:
            if not (logic_problem := sample.get('logic_problem')) or 'parsing_errors' not in logic_problem:
                continue

            error_types = sample.get('error_types', {})
            
            for error, correction in logic_problem["parsing_errors"].items():
                print(f"Error: {error}")
                print(f"Correction: {correction}")
                
                error_type = input("Error type: ").strip()
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
                error_types_round[error_type] = error_types_round.get(error_type, 0) + 1
            
            sample['error_types'] = error_types
            outputs.append(sample)

            self.error_types_global[f"round_{self.config.self_refine_round}"] = error_types_round
            
            FileHandler.save_json(outputs, save_file)
            FileHandler.save_json(self.error_types_global, analysis_file)

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketcher-path', type=str, required=True)
    parser.add_argument('--refiner-path', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--self-refine-round', type=int, required=True)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--save-path', type=str, default='./outputs/annotated')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--gcd', action='store_true')
    args = parser.parse_args()
    return Config(**vars(args))

if __name__ == "__main__":
    config = parse_args()
    engine = ErrorAnnotator(config)
    engine.run()