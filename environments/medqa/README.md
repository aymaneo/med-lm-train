# MedQA

Evaluation environment for the MedQA (USMLE) dataset.

## Overview
- **Environment ID**: `medqa`
- **Short description**: Single-turn USMLE-style medical multiple-choice QA
- **Tags**: medical, single-turn, multiple-choice, usmle, train, eval

## Datasets
- **Primary dataset(s)**: MedQA USMLE 4-options
- **Source links**: [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- **Split sizes**: Uses provided train and test splits

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring (1.0 / 0.0) based on correct letter match

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medqa -m "openai/gpt-5-mini" -n 5 -s --answer-format boxed
```

## PRIME-RL Example
Use this under your PRIME-RL config's `[[orchestrator.env]]` section.

```toml
[[orchestrator.env]]
id = "medqa"
name = "medqa"
args = { use_think = true, train_answer_formats = "random", training_shuffle_answers = true, training_seed = 23 }
```

## Training Behavior
- `training_shuffle_answers = true` enables rollout-time answer-order reshuffling for train rows only.
- The stored training dataset remains in stable order; the presented answer order may change across repeated RL presentations of the same row.
- Eval rows remain fixed unless `shuffle_answers = true` is explicitly set for eval formatting.
- `training_seed` controls training answer-format routing. Rollout-time train reshuffling itself is stochastic.

## Authors
This environment has been put together by:

Ahmed Essouaied - ([@ahmedessouaied](https://github.com/ahmedessouaied))

## Credits
Dataset:

```bibtex
@misc{jin2020diseasedoespatienthave,
      title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
      author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
      year={2020},
      eprint={2009.13081},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2009.13081},
}
```
