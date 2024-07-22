<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">THESIS_PROJECT</h1>
</p>
<p align="center">
    <em>Illuminate logic with power and precision.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/federaspa/thesis_project.git?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/federaspa/thesis_project.git?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/federaspa/thesis_project.git?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/federaspa/thesis_project.git?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

The thesis_project is a sophisticated software endeavor that leverages logic inference, predicates generation, and model evaluation to enhance reasoning and inference tasks. It integrates OpenAI APIs for efficient batch processing and logic program generation, providing a robust framework for logical deductions and model refinement. By automating logical operations and evaluation metrics, the project aims to optimize AI model performance and intelligence, enabling precise reasoning and structured problem-solving capabilities.

---



##  Repository Structure

```sh
└── thesis_project/
    ├── LICENSE
    ├── baselines
    │   ├── evaluation.py
    │   ├── gpt3_baseline.py
    │   ├── icl_examples
    │   │   ├── FOLIO_CoT.txt
    │   │   ├── FOLIO_FewShot.txt
    │   │   └── FOLIO_ZeroShot.txt
    │   ├── results
    │   │   ├── CoT_FOLIO_dev_gpt-3.5-turbo.json
    │   │   ├── CoT_FOLIO_dev_gpt-4o.json
    │   │   ├── FewShot_FOLIO_dev_gpt-3.5-turbo.json
    │   │   ├── FewShot_FOLIO_dev_gpt-4-turbo.json
    │   │   ├── FewShot_FOLIO_dev_gpt-4o.json
    │   │   ├── ZeroShot_FOLIO_dev_gpt-3.5-turbo.json
    │   │   ├── ZeroShot_FOLIO_dev_gpt-4-turbo.json
    │   │   └── ZeroShot_FOLIO_dev_gpt-4o.json
    │   └── utils.py
    ├── models
    │   ├── evaluation.py
    │   ├── examples_extractor.py
    │   ├── gcd_utils.py
    │   ├── grammars
    │   │   ├── FOLIO.gbnf
    │   │   └── LogicNLI.gbnf
    │   ├── infer_all.py
    │   ├── logic_inference.py
    │   ├── logic_predicates.py
    │   ├── logic_program.py
    │   ├── openai_utils.py
    │   ├── prompts
    │   │   ├── FOLIO_dynamic.txt
    │   │   ├── FOLIO_predicates.txt
    │   │   ├── FOLIO_static.txt
    │   │   ├── LogicNLI_dynamic.txt
    │   │   ├── LogicNLI_predicates.txt
    │   │   ├── LogicNLI_static.txt
    │   │   ├── self-correct-execution-FOLIO.txt
    │   │   ├── self-correct-execution-LogicNLI.txt
    │   │   ├── self-correct-parsing-FOLIO.txt
    │   │   └── self-correct-parsing-LogicNLI.txt
    │   ├── refine_all.py
    │   ├── self_refinement.py
    │   ├── symbolic_solvers
    │   │   └── fol_solver
    │   │       ├── Formula.py
    │   │       ├── Formula_util.py
    │   │       ├── __init__.py
    │   │       ├── fol_parser.py
    │   │       ├── fol_prover9_parser.py
    │   │       └── prover9_solver.py
    │   └── task_descriptions
    │       ├── FOLIO.txt
    │       ├── FOLIO_predicates.txt
    │       ├── LogicNLI.txt
    │       ├── LogicNLI_predicates.txt
    │       ├── self-correct-execution-FOLIO.txt
    │       ├── self-correct-execution-LogicNLI.txt
    │       ├── self-correct-parsing-FOLIO.txt
    │       └── self-correct-parsing-LogicNLI.txt
    ├── readme.md
    └── requirements.txt
```

---

##  Modules


| File                                                                                             | Summary                                                                                                                                                             |
| ---                                                                                              | ---                                                                                                                                                                 |
| [requirements.txt](https://github.com/federaspa/thesis_project.git/blob/master/requirements.txt) | Defines crucial dependencies for the project from various libraries via the requirements.txt file. Ensures essential packages are installed for seamless execution. |


<details closed><summary>models</summary>

| File                                                                                                              | Summary                                                                                                                                                                                                                                                                                              |
| ---                                                                                                               | ---                                                                                                                                                                                                                                                                                                  |
| [logic_inference.py](https://github.com/federaspa/thesis_project.git/blob/master/models/logic_inference.py)       | Generates logic inferences from dataset programs, executing logic programs safely and creating corresponding outputs. Handles parsing and execution errors, saving results, and cleaning up after processing. Controlled by configurable arguments.                                                  |
| [logic_predicates.py](https://github.com/federaspa/thesis_project.git/blob/master/models/logic_predicates.py)     | Generates logic predicates based on problem description for the given dataset, utilizing OpenAI API for batch or async processing. Handles prompt templates, parsing, and dataset processing for logic program generation.                                                                           |
| [evaluation.py](https://github.com/federaspa/thesis_project.git/blob/master/models/evaluation.py)                 | Calculates and analyzes evaluation metrics on model predictions for a given dataset. Handles precision, recall, F1 scores, and more to measure model performance. Key insights are provided on executable rates, errors, and detailed breakdowns of metrics.                                         |
| [infer_all.py](https://github.com/federaspa/thesis_project.git/blob/master/models/infer_all.py)                   | Coordinates logic inference for different sketchers, refiners, and load directories based on specified arguments, facilitating refinement and program inference. Manages self-refinement rounds and leverages a configurable engine for dataset inference.                                           |
| [gcd_utils.py](https://github.com/federaspa/thesis_project.git/blob/master/models/gcd_utils.py)                   | Enables model invocation with grammatical constraints for generating responses based on user input and task descriptions within a specified context length and token generation limit. The model integrates a LlamaGrammar object to facilitate accurate language completion.                        |
| [refine_all.py](https://github.com/federaspa/thesis_project.git/blob/master/models/refine_all.py)                 | Refines and refactors AI models using specified sketchers and refiners. Implements self-refinement engine for multiple rounds, handling exceptions and user interrupts. Impacts the models performance and accuracy in the thesis project architecture.                                              |
| [examples_extractor.py](https://github.com/federaspa/thesis_project.git/blob/master/models/examples_extractor.py) | Extracts and organizes similar examples from given datasets using OpenAIs text embeddings. Utilizes cosine similarity to rank examples and minimize duplicates. Saves results in a JSON file based on specified criteria.                                                                            |
| [openai_utils.py](https://github.com/federaspa/thesis_project.git/blob/master/models/openai_utils.py)             | Implements OpenAI API interactions for chat completion and batch requests. Handles exponential backoff, asynchronous dispatch, and model-specific responses within the thesis projects AI model architecture. Handles API rate limits and efficiently manages batch requests for varied model names. |
| [logic_program.py](https://github.com/federaspa/thesis_project.git/blob/master/models/logic_program.py)           | Generates prompts for logic programs based on provided datasets, allowing dynamic or static modes. Supports batch generation efficiency with synchronous or asynchronous OpenAI API calls. Optionally, cheats logic program generation for unreleased feature.                                       |
| [self_refinement.py](https://github.com/federaspa/thesis_project.git/blob/master/models/self_refinement.py)       | Generates refined logic programs for datasets by executing self-refinement rounds. Loads logic programs, predicates, and ground truth. Utilizes OpenAI for generation and handles parsing and execution errors, updating programs accordingly in each iteration.                                     |

</details>

<details closed><summary>models.symbolic_solvers.fol_solver</summary>

| File                                                                                                                                          | Summary                                                                                                                                                                                                                                                                      |
| ---                                                                                                                                           | ---                                                                                                                                                                                                                                                                          |
| [Formula.py](https://github.com/federaspa/thesis_project.git/blob/master/models/symbolic_solvers/fol_solver/Formula.py)                       | Implements FOL formula parsing and transformation; signals timeout and extracts template for logic formulas in the parent repositorys symbolic solvers module.                                                                                                               |
| [prover9_solver.py](https://github.com/federaspa/thesis_project.git/blob/master/models/symbolic_solvers/fol_solver/prover9_solver.py)         | Validates, parses, and executes logical programs using Prover9 solver for first-order logic questions. Handles premises, conclusions, and NL statements to determine provability, returning True, False, or Unknown. Converting logic rules to Prover9 format for inference. |
| [fol_prover9_parser.py](https://github.com/federaspa/thesis_project.git/blob/master/models/symbolic_solvers/fol_solver/fol_prover9_parser.py) | Transforms logical formula structure for Prover9 validation. Implements grammar adjustments for reliable inference. Parses FOL formulas into a Prover9-compatible format, enhancing logic processing for the repositorys symbolic solvers model.                             |
| [Formula_util.py](https://github.com/federaspa/thesis_project.git/blob/master/models/symbolic_solvers/fol_solver/Formula_util.py)             | Defines a Class for parsing FOL formulas, ensuring validity by resolving symbols. Implements a method to generate a template from the parsed formula, mapping symbols accordingly. Executable snippet showcases parsing and template extraction functionalities.             |
| [fol_parser.py](https://github.com/federaspa/thesis_project.git/blob/master/models/symbolic_solvers/fol_solver/fol_parser.py)                 | Parses and transforms First-Order Logic rules into an nltk.tree, resolving symbols into variables, constants, and predicates for further processing within the repositorys symbolic solvers module.                                                                          |

</details>

<details closed><summary>models.grammars</summary>

| File                                                                                                       | Summary                                                                                                                                                                                                           |
| ---                                                                                                        | ---                                                                                                                                                                                                               |
| [FOLIO.gbnf](https://github.com/federaspa/thesis_project.git/blob/master/models/grammars/FOLIO.gbnf)       | Quantifiers, logical connectives, predicates, and variables. Supports complex logical expressions, enabling inference on a symbolic level within the repositorys model architecture.                              |
| [LogicNLI.gbnf](https://github.com/federaspa/thesis_project.git/blob/master/models/grammars/LogicNLI.gbnf) | Defines logic grammar for LogicNLI in the models module. Contains rules for logical expressions using quantifiers, logical operators, and predicates for automated reasoning within the repositorys architecture. |

</details>

<details closed><summary>models.task_descriptions</summary>

| File                                                                                                                                                            | Summary                                                                                                                                                                                                                                                      |
| ---                                                                                                                                                             | ---                                                                                                                                                                                                                                                          |
| [LogicNLI.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/LogicNLI.txt)                                               | Generates First-Order-Logic rules and questions from Natural Language Problems and Questions using specified predicates and operators. Maintains strict rules for accurate and valid conversion while outputting results in a predefined JSON format.        |
| [LogicNLI_predicates.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/LogicNLI_predicates.txt)                         | Generates first-order logic predicates from natural language problems, ensuring each predicate includes its corresponding meaning. Prohibits having predicate names within others. Mandates responses in valid JSON format with predicate and meaning pairs. |
| [self-correct-parsing-FOLIO.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/self-correct-parsing-FOLIO.txt)           | Improve First Order Logic statements for accuracy and consistency according to provided predicates in Natural Language processing tasks, ensuring correct usage of logical operators and adhering to specific syntax rules.                                  |
| [FOLIO.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/FOLIO.txt)                                                     | Defines strict rules for converting natural language problems to First-Order-Logic with specific predicates and operators. Ensures accurate reflection of meaning and valid JSON response generation in a specified format.                                  |
| [self-correct-execution-LogicNLI.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/self-correct-execution-LogicNLI.txt) | Identifies errors, provides fixes, and generates corrected First-Order-Logic rules based on given examples and constraints.                                                                                                                                  |
| [self-correct-parsing-LogicNLI.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/self-correct-parsing-LogicNLI.txt)     | Refines Sketch FOL statements using specified predicates, logical operators, and guidelines, correcting errors while ensuring accuracy and proximity to the original sketch.                                                                                 |
| [self-correct-execution-FOLIO.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/self-correct-execution-FOLIO.txt)       | Identifies errors, proposes fixes, and generates corrected First-Order Logic rules and questions in structured JSON format. Ensures adherence to specific logical operators, predicates, and entity naming conventions.                                      |
| [FOLIO_predicates.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/task_descriptions/FOLIO_predicates.txt)                               | Generates First-Order-Predicates with Natural Language explanations based on examples, ensuring unique predicate names. Outputs in JSON format within task_descriptions for the FOLIO project.                                                               |

</details>

<details closed><summary>models.prompts</summary>

| File                                                                                                                                                  | Summary                                                                                                                                                                                                                                                                               |
| ---                                                                                                                                                   | ---                                                                                                                                                                                                                                                                                   |
| [LogicNLI_predicates.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/LogicNLI_predicates.txt)                         | Implements first-order logic predicates for natural language problems in LogicNLI. Describes characteristics like blue, serious, fresh, entire, accurate, concerned. Replicates scenarios with named individuals to create logic statements.                                          |
| [FOLIO_dynamic.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/FOLIO_dynamic.txt)                                     | Generates structured prompts for First-Order-Logic (FOL) problems. Defines natural language problems, questions, and FOL predicates for each prompt. Supports FOL rule and question handling, enhancing logic-based problem-solving capabilities within the repositorys architecture. |
| [LogicNLI_dynamic.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/LogicNLI_dynamic.txt)                               | Generates structured prompts for Natural Language Inference tasks based on predefined problems, questions, predicates, rules, and logic components.                                                                                                                                   |
| [self-correct-parsing-FOLIO.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/self-correct-parsing-FOLIO.txt)           | Generates first-order logic statements from natural language descriptions, mapping predicates for symbolic reasoning. Clarifies ambiguous statements and ensures logical consistency for structured reasoning tasks in the AI project.                                                |
| [self-correct-execution-LogicNLI.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/self-correct-execution-LogicNLI.txt) | Correcting predicate order to prevent errors and substituting invalid logic operator with a new predicate for accurate logical representation.                                                                                                                                        |
| [LogicNLI_static.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/LogicNLI_static.txt)                                 | Analyzes natural language text, defining predicates and rules for logical inference in a symbolic solver. Captures relationships to deduce conclusions based on given information and predicates, forming a complex logic puzzle scenario.                                            |
| [FOLIO_static.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/FOLIO_static.txt)                                       | Defines natural language problems and questions using first-order logic predicates for symbolic reasoning in the thesis projects AI model development.                                                                                                                                |
| [self-correct-parsing-LogicNLI.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/self-correct-parsing-LogicNLI.txt)     | Transforms natural language into logical FOL statements for LogicNLI prompt examples, defining predicates and valid FOL statements. Enhances understanding and reasoning capabilities in the LogicNLI task.                                                                           |
| [self-correct-execution-FOLIO.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/self-correct-execution-FOLIO.txt)       | Ensures logical consistency by fixing errors in predicates and operators. Maintains integrity and accuracy in reasoning tasks within the FOLIO domain.                                                                                                                                |
| [FOLIO_predicates.txt](https://github.com/federaspa/thesis_project.git/blob/master/models/prompts/FOLIO_predicates.txt)                               | Defines essential First-Order-Logic predicates for natural language problems surrounding dependencies, student attributes, and music specialization within the repositorys logic-based inference models.                                                                              |

</details>

<details closed><summary>baselines</summary>

| File                                                                                                       | Summary                                                                                                                                                                                                                                                                           |
| ---                                                                                                        | ---                                                                                                                                                                                                                                                                               |
| [evaluation.py](https://github.com/federaspa/thesis_project.git/blob/master/baselines/evaluation.py)       | Calculates evaluation metrics for question answering results. Parses answers, evaluates precision, recall, and F1 scores, and provides summary statistics. Handles backup data retrieval and assesses parsing and execution errors. Configurable for various datasets and models. |
| [gpt3_baseline.py](https://github.com/federaspa/thesis_project.git/blob/master/baselines/gpt3_baseline.py) | Generates reasoning graphs from input data, leveraging an OpenAI model for prediction. Implements batch processing for efficiency. Provenance and predictions saved based on dataset split and model used, amplifying project efficiency.                                         |
| [utils.py](https://github.com/federaspa/thesis_project.git/blob/master/baselines/utils.py)                 | Enables asynchronous requests to OpenAIs ChatCompletion API with backoff retries. Provides methods for generating responses using various OpenAI models and handling batches of messages efficiently. Supports dynamic model selection based on the designated model name.        |

</details>

<details closed><summary>baselines.icl_examples</summary>

| File                                                                                                                        | Summary                                                                                                                                                                                                                                              |
| ---                                                                                                                         | ---                                                                                                                                                                                                                                                  |
| [FOLIO_ZeroShot.txt](https://github.com/federaspa/thesis_project.git/blob/master/baselines/icl_examples/FOLIO_ZeroShot.txt) | Presents logical reasoning questions with contexts, requiring answers in a specific format. Contexts dictate problem statements, followed by a question and answer options. Designed to prompt users to provide answers precisely as single letters. |
| [FOLIO_CoT.txt](https://github.com/federaspa/thesis_project.git/blob/master/baselines/icl_examples/FOLIO_CoT.txt)           | Validates logical reasoning answers from given contexts in JSON format, including reasoning and answer. Engages users with problem statements and questions to assess logical deductions.                                                            |
| [FOLIO_FewShot.txt](https://github.com/federaspa/thesis_project.git/blob/master/baselines/icl_examples/FOLIO_FewShot.txt)   | Analyzes logical reasoning tasks with problem contexts and questions, providing multiple-choice options and correct answers. Enables evaluating if statements are true, false, or uncertain based on given information.                              |

</details>

<details closed><summary>baselines.results</summary>

| File                                                                                                                                                         | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---                                                                                                                                                          | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [CoT_FOLIO_dev_gpt-4o.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/CoT_FOLIO_dev_gpt-4o.json)                         | Thesis_project/models/logic_inference.py`The `logic_inference.py` file in the `thesis_project` repository plays a crucial role in performing logical reasoning and inference tasks within the projects architecture. It serves as a central component for handling complex logic-based computations and making deductions based on predefined rules and grammars. By leveraging this file, the project can execute sophisticated logic-based operations, enhancing the overall functionality and intelligence of the models used in the system.      |
| [FewShot_FOLIO_dev_gpt-3.5-turbo.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/FewShot_FOLIO_dev_gpt-3.5-turbo.json)   | Analyzes model predictions on questions in a JSON file. Verifies answers, reasoning, and predictions for various scenarios. Facilitates assessment of model performance in reasoning and comprehension tasks within the repositorys NLP baselines.                                                                                                                                                                                                                                                                                                   |
| [FewShot_FOLIO_dev_gpt-4o.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/FewShot_FOLIO_dev_gpt-4o.json)                 | The `evaluation.py` file within the `models` directory provides essential functions for evaluating the performance of models developed in the `thesis_project` repository. It offers functionality to measure accuracy, precision, and recall metrics for various tasks, aiding in the assessment of model effectiveness. This code file serves as a crucial component for assessing the efficacy of models trained within the architecture of the parent repository.                                                                                |
| [FewShot_FOLIO_dev_gpt-4-turbo.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/FewShot_FOLIO_dev_gpt-4-turbo.json)       | The `gpt3_baseline.py` file within the `baselines` directory of the `thesis_project` repository serves as a critical component for evaluating and generating results using the GPT-3 baseline model. This code file encapsulates the logic for leveraging the GPT-3 model to perform specific tasks and analyze the results. It plays a pivotal role in the repositorys architecture by providing a standardized approach for baseline evaluations and result generation within the context of the broader research and experimentation environment. |
| [ZeroShot_FOLIO_dev_gpt-4-turbo.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/ZeroShot_FOLIO_dev_gpt-4-turbo.json)     | Analyzes reasoning and answers for various questions. Predicts logical outcomes based on given premises. Supports decision-making using structured reasoning. Supports academic, logical, and predictive tasks within the machine learning context.                                                                                                                                                                                                                                                                                                  |
| [ZeroShot_FOLIO_dev_gpt-4o.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/ZeroShot_FOLIO_dev_gpt-4o.json)               | Analyzes and predicts outcomes for various scenarios within a JSON file containing question-answer pairs. Aids in assessing logical reasoning and answer prediction using the provided data.                                                                                                                                                                                                                                                                                                                                                         |
| [CoT_FOLIO_dev_gpt-3.5-turbo.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/CoT_FOLIO_dev_gpt-3.5-turbo.json)           | The `gpt3_baseline.py` file in the `baselines` directory of the repository serves as a crucial component of the thesis project. It implements a baseline model leveraging GPT-3 for specific tasks. This code file contributes to the evaluation of different models using GPT-3 across various examples within the projects architecture.                                                                                                                                                                                                           |
| [ZeroShot_FOLIO_dev_gpt-3.5-turbo.json](https://github.com/federaspa/thesis_project.git/blob/master/baselines/results/ZeroShot_FOLIO_dev_gpt-3.5-turbo.json) | Summarizes logic predictions for various statements. Categorizes answers as True, False, or Undetermined. Facilitates reasoning assessment.                                                                                                                                                                                                                                                                                                                                                                                                          |

</details>

---

##  Getting Started

**System Requirements:**

* **Text**: `version 3.9.19`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the thesis_project repository:
>
> ```console
> $ git clone https://github.com/federaspa/thesis_project.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd thesis_project
> ```
>
> 3. Install the dependencies:
> ```console
> $ > pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run thesis_project using the command below:
> ```console
> $ 
> ```

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-overview)

---
