from typing import Tuple, Optional
import yaml
from pathlib import Path
import re
from dataclasses import dataclass
import argparse


@dataclass
class ScriptConfig:
    model_name: str
    max_model_size: str
    min_model_size: str
    dataset_name: str
    shots_number: str
    data_path: str
    split: str
    models_path: str
    save_path: str
    timeout: int
    starting_sample: int
    verbose: bool
    n_gpu_layers: int
    start_time: Optional[str]
    stop_time: Optional[str]
    force_unconstrained: Optional[bool]
    force_constrained: Optional[bool]
    force_json: Optional[bool]
    
def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-name', type=str, default=None, help='Filter models by name')
    parser.add_argument('--max-model-size', type=str, default=None, help='Maximum model size (e.g., "7b", "14b")')
    parser.add_argument('--min-model-size', type=str, default=None, help='Minimum model size (e.g., "7b", "14b")')
    parser.add_argument('--shots-number', type=str, default='2shots', choices=['0shots', '2shots', '5shots', 'baseline'])
    parser.add_argument('--n-gpu-layers', type=int, default=-1)
    parser.add_argument('--dataset-name', type=str, default='FOLIO')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--models-path', type=str, default='/home/fraspanti/LLMs')
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('--save-path', type=str, default='outputs')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for generation operations')
    parser.add_argument('--start-time', default=None, type=str, help='Start time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--stop-time', default=None, type=str, help='Stop time in format dd-mm-yy:hh-mm-ss')
    parser.add_argument('--starting-sample', type=int, default=0)
    parser.add_argument('--force-unconstrained', action='store_true')
    parser.add_argument('--force-constrained', action='store_true')
    parser.add_argument('--force-json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    return ScriptConfig(**vars(parser.parse_args()))
    
def get_configs(script_config: ScriptConfig) -> Tuple[ScriptConfig, dict, dict]:
    full_model_name = re.sub(r'-\d+-of-\d+', '', script_config.model_name)
    
    # Load LlamaCpp generation configuration which now contains all needed parameters
    llamacpp_config_path = Path('configs') / 'llamacpp.yml'
    with open(llamacpp_config_path, 'r') as f:
        llama_cpp_config = yaml.safe_load(f)
    
    # Extract model parameters from the llama_cpp_config
    model_config = llama_cpp_config['model_parameters']
    # Update llama_cpp_config to only contain the llama_cpp parameters
    llama_cpp_parameters = llama_cpp_config['llama_cpp_parameters']
    
    # Add model path to config
    base_model_name = full_model_name.split('-')[0]
    model_dir = Path(script_config.models_path) / base_model_name
    
    # Check for multi-part model files
    model_pattern = f"{script_config.model_name}-*-of-*.gguf"
    model_parts = list(model_dir.glob(model_pattern))
    
    if model_parts:
        # Multi-part model - find first part
        model_parts.sort()  # Ensure proper ordering
        model_path = str(model_parts[0])
    else:
        # Single file model
        model_path = str(model_dir / f"{script_config.model_name}.gguf")
        
    # Verify model path exists
    assert Path(model_path).exists(), f'Model not found: {model_path}'
    
    # Update llama_cpp parameters with command line arguments
    llama_cpp_parameters['model_path'] = model_path
    llama_cpp_parameters['verbose'] = script_config.verbose
    llama_cpp_parameters['n_gpu_layers'] = script_config.n_gpu_layers
    llama_cpp_parameters['n_threads'] = llama_cpp_parameters.get('n_threads', 4) if script_config.n_gpu_layers == -1 else 20
    
    return script_config, model_config, llama_cpp_parameters

def extract_model_size(model_name):
    # Extract the size part (e.g., "8b" from "modelname2.0-8b-it")
    try:
        # Find the part that ends with 'b' and convert to number
        size_str = next(part for part in model_name.split('-') if part.endswith('b'))
        return int(size_str[:-1])  # Remove 'b' and convert to int
    except (StopIteration, ValueError):
        return 0  # Return 0 if pattern not found or conversion fails

def get_models(models_path: str, model_name_filter: str = None, max_model_size: str = None, min_model_size: str = None) -> list:
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
    
    # Convert max_model_size to integer if provided
    max_size = None
    min_size = None
    if max_model_size:
        if max_model_size.endswith("b"):
            try:
                max_size = int(max_model_size[:-1])
            except ValueError:
                pass
            
    if min_model_size:
        if min_model_size.endswith("b"):
            try:
                min_size = int(min_model_size[:-1])
            except ValueError:
                pass
            
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
            # Apply name filter
            if model_name_filter is not None and model_name_filter not in model_name:
                continue
                
            # Apply size filter
            if max_size is not None:
                model_size = extract_model_size(model_name)
                if model_size > max_size:
                    continue
            if min_size is not None:
                model_size = extract_model_size(model_name)
                if model_size < min_size:
                    continue
                    
            models.append(model_name)
                
        # Add multi-part models
        for model_name in multipart_models:
            # Apply name filter
            if model_name_filter is not None and model_name_filter not in model_name:
                continue
                
            # Apply size filter
            if max_size is not None:
                model_size = extract_model_size(model_name)
                if model_size > max_size:
                    continue
            if min_size is not None:
                model_size = extract_model_size(model_name)
                if model_size < min_size:
                    continue
                    
            models.append(model_name)
    
    # Sort models by size (largest first)
    return sorted(models, key=extract_model_size, reverse=True)