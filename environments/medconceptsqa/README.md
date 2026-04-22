# medconceptsqa

## Overview
- **Environment ID**: `medconceptsqa`
- **Short description**: MedConcepts QA - an MCQ dataset involving medical codes.
- **Tags**: medical, clinical, single-turn, multiple-choice, classification, test

## Datasets
- **Primary dataset(s)**: `medconceptsqa`
- **Source links**: [Paper](https://www.sciencedirect.com/science/article/pii/S0010482524011740), [Github](https://github.com/nadavlab/MedConceptsQA/tree/master), [HF Dataset](https://huggingface.co/datasets/ofir408/MedConceptsQA)
- **Split sizes**: 60 (dev / few-shot), 820k (test)

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correct answer choice

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medconceptsqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medconceptsqa -m "openai/gpt-5-mini" -n 20 --num-few-shot 4
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_few_shot` | int | `0` | Number of few-shot examples to include in the prompt |
| `use_think` | bool | `False` | Whether to use `<think>...</think>` formatting with `ThinkParser` |
| `train_answer_formats` | string or list | eval format | Training answer format routing; `"random"` cycles across xml/boxed/json by row |
| `training_shuffle_answers` | bool | `false` | Reshuffle answer order at rollout time for train rows only |
| `training_seed` | int | `1618` | Seed for training answer-format routing |
| `shuffle_answers` | bool | `false` | Eval-only answer shuffling |

## Training Behavior
- `training_shuffle_answers = true` reshuffles answer order at rollout time for train rows only.
- The stored training dataset remains stable; repeated RL presentations of the same row may use different answer orders.
- Few-shot exemplars remain fixed; only the target question is reshuffled at rollout time.
- Eval rows remain fixed unless `shuffle_answers = true` is explicitly enabled.
- `training_seed` controls training answer-format routing. Rollout-time train reshuffling itself is stochastic.

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | Exact match on target answer |

## Authors
This environment has been put together by:

Anish Mahishi - ([@macandro96](https://github.com/macandro96))
