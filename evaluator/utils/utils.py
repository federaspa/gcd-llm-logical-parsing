from pathlib import Path
import re

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