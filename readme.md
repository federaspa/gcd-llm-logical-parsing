##  Getting Started

**System Requirements:**

* **Python**: `version 3.11`

###  Installation

<h4>From <code>source</code></h4>

1. Clone the thesis_project repository:
>
> ```console
> $ git clone https://github.com/federaspa/thesis_project.git
> ```
>
2. Change to the project directory:
> ```console
> $ cd thesis_project
> ```
>
3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```
4. Download the Q6_K.gguf version of the desired open-source LLM for refinement in `GCD/llms`, for example
>```console
>$ GCD/llms/llama-2-7b.Q6_K.gguf
>```
4. Download and compile Prover9 into `models/symbolic_solvers/Prover9`, following https://www.cs.unm.edu/~mccune/prover9/manual/2009-11A/

###  Usage

<h4>From <code>source</code></h4>

1. Extract examples from train split
> ```console
> $ python models/examples_extractor.py --dataset_name FOLIO
> ```
2. Extract predicates for CFG grammar update
> ```console
> python models/logic_predicates.py --dataset_name FOLIO
> ```
3. Extract logic programs
> ```console
> python models/logic_program.py --dataset_name FOLIO
> ```
4. Refine logic programs. Pass `--gcd` to make use of grammar-constrained decoding. To use GPU acceleartion, set `--n-gpu-layers` to a number greater than 0, or to `-1` to pass all model's layers to the GPU.
> ```console
> python models/self_refinement.py --dataset_name FOLIO  --maximum_rounds 3 --gcd --n_gpu_layers 0
> ```
5. Run Prover9 inference 
> ```console
> python models/logic_inference.py --dataset_name FOLIO  --self_refine_round 3 
> ```
