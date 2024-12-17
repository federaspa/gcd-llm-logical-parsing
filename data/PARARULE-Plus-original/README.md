---
license: mit
task_categories:
- text-classification
- question-answering
language:
- en
tags:
- Reasoning
- Multi-Step-Deductive-Reasoning
- Logical-Reasoning
size_categories:
- 100K<n<1M
---
# PARARULE-Plus
This is a branch which includes the dataset from PARARULE-Plus Depth=2, Depth=3, Depth=4 and Depth=5. PARARULE Plus is a deep multi-step reasoning dataset over natural language. It can be seen as an improvement on the dataset of PARARULE (Peter Clark et al., 2020). Both PARARULE and PARARULE-Plus follow the closed-world assumption and negation as failure. The motivation is to generate deeper PARARULE training samples. We add more training samples for the case where the depth is greater than or equal to two to explore whether Transformer has reasoning ability. PARARULE Plus is a combination of two types of entities, animals and people, and corresponding relationships and attributes. From the depth of 2 to the depth of 5, we have around 100,000 samples in the depth of each layer, and there are nearly 400,000 samples in total.

Here is the original links for PARARULE-Plus including paper, project and data.

Paper: https://www.cs.ox.ac.uk/isg/conferences/tmp-proceedings/NeSy2022/paper15.pdf

Project: https://github.com/Strong-AI-Lab/Multi-Step-Deductive-Reasoning-Over-Natural-Language

Data: https://github.com/Strong-AI-Lab/PARARULE-Plus

PARARULE-Plus has been collected and merged by [LogiTorch.ai](https://www.logitorch.ai/), [ReasoningNLP](https://github.com/FreedomIntelligence/ReasoningNLP), [Prompt4ReasoningPapers](https://github.com/zjunlp/Prompt4ReasoningPapers) and [OpenAI/Evals](https://github.com/openai/evals/pull/651).

In this huggingface version, we pre-processed the dataset and use `1` to represent `true` and `0` to represent `false` to better help user train model.

## How to load the dataset?
```
from datasets import load_dataset
dataset = load_dataset("qbao775/PARARULE-Plus")
```

## How to train a model using the dataset?

We provide an [example](https://github.com/Strong-AI-Lab/PARARULE-Plus/blob/main/README.md#an-example-script-to-load-pararule-plus-and-fine-tune-bert) that you can `git clone` the project and fine-tune the dataset locally.

## Citation
```
@inproceedings{bao2022multi,
  title={Multi-Step Deductive Reasoning Over Natural Language: An Empirical Study on Out-of-Distribution Generalisation},
  author={Qiming Bao and Alex Yuxuan Peng and Tim Hartill and Neset Tan and Zhenyun Deng and Michael Witbrock and Jiamou Liu},
  year={2022},
  publisher={The 2nd International Joint Conference on Learning and Reasoning and 16th International Workshop on Neural-Symbolic Learning and Reasoning (IJCLR-NeSy 2022)}
}
```