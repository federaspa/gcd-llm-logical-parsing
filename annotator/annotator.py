import streamlit as st
import json
import random
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class NLProblem:
    context: List[str]
    question: str
    options: List[str]

@dataclass
class LogicProblem:
    fol_preds: List[str]
    fol_consts: List[str]
    fol_rules: List[str]
    fol_conc: str
    perplexity: Optional[List[float]]
    status: Optional[str]
    error: Optional[str]

def load_samples(file_path: str) -> List[dict]:
    """Load prepared samples from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_annotation(annotation: dict, output_file: str):
    """Save annotation to JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:
        json.dump(annotation, f)
        f.write('\n')

def prepare_samples(experiment_files: List[str], samples_per_file: int, output_path: str):
    """Prepare random samples from multiple experiment files, with each sample duplicated 3 times."""
    samples = []
    for file_path in experiment_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            data = [d for d in data 
                    if 'logic_problem' in d.keys() 
                    or 'logic_problem_gcd' in d.keys() 
                    or 'logic_problem_twosteps' in d.keys()]
            for item in data:
                item['source_file'] = os.path.basename(file_path)
            file_samples = random.sample(data, min(samples_per_file, len(data)))
            # Create 3 copies of each sample
            duplicated_samples = file_samples * 3
            samples.extend(duplicated_samples)
    
    random.shuffle(samples)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    return len(samples)

def display_nl_problem(problem: dict):
    """Display the natural language problem."""
    st.header("Natural Language Problem")
    st.subheader("Context")
    for idx, premise in enumerate(problem['context'], 1):
        st.write(f"{idx}. {premise}")
    st.subheader("Question")
    st.write(problem['question'])

def display_logic_problem(version_name: str, problem: dict, nl_context_length: int):
    """Display a logic problem version and its annotation interface."""
    st.header(f"{version_name}")

    st.subheader("Predicates")
    st.write(", ".join(problem['fol_preds']))

    pred_sound = st.checkbox("Sound", key=f"{version_name}_pred_sound")
    pred_complete = st.checkbox("Complete", key=f"{version_name}_pred_complete")
        
    st.subheader("Constants")
    st.write(", ".join(problem['fol_consts']))

    const_sound = st.checkbox("Sound", key=f"{version_name}_const_sound")
    const_complete = st.checkbox("Complete", key=f"{version_name}_const_complete")

    st.subheader("Rules")
    rule_checks = []
    rules = [r for r in problem['fol_rules'] if r]
    for idx, rule in enumerate(rules):
        rule_checks.append(st.checkbox(f"- {rule}", key=f"{version_name}_rule_{idx}", label_visibility="visible"))
    
    st.subheader("Conclusion")
    conc_check = st.checkbox(f"- {problem['fol_conc']}", key=f"{version_name}_conclusion", label_visibility="visible")
        
    out = {
        'predicates': {
            'sound': pred_sound,
            'complete': pred_complete
        },
        'constants': {
            'sound': const_sound,
            'complete': const_complete
        },
        'rules': rule_checks,
        'conclusion': conc_check
    }
    
    return out

def main():
    st.set_page_config(page_title="Experiment Annotation Tool", layout="wide")
    st.title("Experiment Annotation Tool")

    # Session state initialization
    if 'current_sample_idx' not in st.session_state:
        st.session_state.current_sample_idx = 0
    if 'samples' not in st.session_state:
        st.session_state.samples = []

    with st.sidebar:
        st.header("Configuration")
        
        session_type = st.radio("Session Type", ["Resume Previous", "New Session"])
        
        if session_type == "Resume Previous" and os.path.exists("prepared_samples.json"):
            if st.button("Load Previous Session"):
                st.session_state.samples = load_samples("prepared_samples.json")
                # Find last annotated sample
                if os.path.exists("annotations/annotations.jsonl"):
                    with open("annotations/annotations.jsonl", 'r') as f:
                        annotations = [json.loads(line) for line in f]
                        if annotations:
                            last_sample_id = annotations[-1]['sample_id']
                            for idx, sample in enumerate(st.session_state.samples):
                                if sample['id'] == last_sample_id:
                                    st.session_state.current_sample_idx = idx + 1
                                    break
        else:
            # Modified sample preparation interface
            st.subheader("Prepare Samples")
            samples_per_file = st.number_input("Samples per file", min_value=1, value=50)
            if st.button("Prepare Samples"):
                # Get all JSON files from data directory
                data_dir = 'data'
                experiment_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                                 if f.endswith('.json')]
                
                if not experiment_files:
                    st.error("No JSON files found in the 'data' directory!")
                else:
                    count = prepare_samples(experiment_files, samples_per_file, "prepared_samples.json")
                    st.session_state.samples = load_samples("prepared_samples.json")
                    st.success(f"Prepared {count} samples from {len(experiment_files)} files")

        # Progress tracking
        if st.session_state.samples:
            total_samples = len(st.session_state.samples)
            remaining = total_samples - st.session_state.current_sample_idx
            st.write(f"Total samples: {total_samples}")
            st.write(f"Remaining: {remaining}")
            st.write(f"Current: {st.session_state.current_sample_idx + 1}/{total_samples}")

    # Rest of the main function remains the same
    if not st.session_state.samples:
        st.info("Please prepare samples using the sidebar configuration.")
        return

    current_sample = st.session_state.samples[st.session_state.current_sample_idx]
    
    display_nl_problem(current_sample['nl_problem'])
    
    st.write("---")
    annotations = {}
    
    versions = list(k for k in current_sample.keys() if 'logic_problem' in k)
    
    for i, version in enumerate(versions):
        version_display = f"Version {i+1}"
        annotations[version] = display_logic_problem(
            version_display,
            current_sample[version],
            len(current_sample['nl_problem']['context'])
        )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous") and st.session_state.current_sample_idx > 0:
            st.session_state.current_sample_idx -= 1
            st.rerun()
    
    with col2:
        if st.button("Next/Submit"):
            annotation = {
                'sample_id': current_sample['id'],
                'source_file': current_sample['source_file'],
                'annotations': annotations,
                'timestamp': datetime.now().isoformat(),
            }
            save_annotation(annotation, "annotations/annotations.jsonl")
            
            if st.session_state.current_sample_idx < len(st.session_state.samples) - 1:
                st.session_state.current_sample_idx += 1
                st.rerun()
            else:
                st.success("All samples annotated!")
if __name__ == "__main__":
    main()