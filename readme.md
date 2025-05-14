# Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers

This repository contains the code for the paper "Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers."

## System Requirements

- **Python**: version 3.11
- **Dependencies**: Install with `pip install -r requirements.txt`

## Supported Models

The following LLM families are supported:
- `gemma2`
- `llama3.1`
- `llama3.2` 
- `mistral`
- `ministral`
- `qwen2.5`

## Setup and Reproduction

### 1. Download Models

Download GGUF versions of any supported LLM and place them in an accessible directory. The file structure should be:

```
[model_directory]/
├── gemma2/
│   ├── gemma2-2b.gguf
│   ├── gemma2-9b.gguf
│   └── gemma2-27b.gguf
├── llama3.1/
│   ├── llama3.1-8b.gguf
│   └── llama3.1-70b.gguf
└── ...
```

### 2. Install Symbolic Solvers

For FOLIO dataset evaluation, Prover9 is required:
1. Download and compile Prover9 from https://www.cs.unm.edu/~mccune/prover9/manual/2009-11A/
2. Place the compiled binaries in `scripts/symbolic_solvers/Prover9/`

### 3. Generate Logical Programs

Run the `logic_problems.py` script to generate logical representations from natural language problems:

```bash
python scripts/logic_problems.py --dataset_name FOLIO --shots_number 2shots --model_name [model_name] --models_path [path_to_models]
```

Options:
- `--shots_number`: Choose from `0shots`, `2shots`, `5shots`, `baseline`
- `--dataset_name`: Set to `FOLIO` for first-order logic or `GSM8K_symbolic` for arithmetic tasks
- `--model_name`: Specific model to use (e.g., `gemma2-9b`)
- `--models_path`: Directory containing the downloaded models

### 4. Run Symbolic Inference

Process the generated logical representations through symbolic solvers:

```bash
python scripts/logic_inference.py --dataset_name FOLIO --shots_number 2shots --model_name [model_name]
```

### 5. Evaluate Results

Calculate accuracy and coverage metrics for the inference results:

```bash
python evaluator/evaluation.py --result-path outputs --model_name [model_name] --dataset_name FOLIO --split dev
```

Additional options:
- `--save-df`: Save results as a dataframe
- `--latex`: Output results in LaTeX table format

## Citation

If you use this code for your research, please cite our paper:
```
[Citation information will be added after publication]
```
