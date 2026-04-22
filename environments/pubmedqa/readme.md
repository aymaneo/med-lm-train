# pubmedqa environment

### Overview
- **Environment ID**: `pubmedqa`
- **Short description**: pubmedqa test set from Jin et al. 2019

### Dataset

- **Primary dataset(s)**: PubMedQA 1k expert-annotated QA instances (500 instances test subset)

The 1k instances are available directly on the [pubmedqa githib repoisitory](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json), and is available on the huggingface hub as ['qiaojin/PubMedQA'](https://huggingface.co/datasets/qiaojin/PubMedQA), which is also the dataset used by InspectEval.

The IDs of the 500 test instances are mapped in [test_ground_truth.json](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json).

This should correspond to the 'pqal_test_set.json' (from the bigbio/pubmed_qa [pqal.zip](https://huggingface.co/datasets/bigbio/pubmed_qa/blob/main/pqal.zip) file), resulting after [splitting](https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py) the 1k set.


### Task
- **Type**: single-turn
- **Parser**: InspectEval-style MCQ parser
- **Rubric**: Classification-based rubric (1 point for correct answer, 0 points for incorrect answer). Three choices: (yes/no/maybe)

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run pubmedqa -m "openai/gpt-5-mini" -n 5 -s
```

Or use `medarc-eval` for named flags:

```bash
medarc-eval pubmedqa -m "openai/gpt-5-mini" -n 5 -s --shuffle-answers --shuffle-seed 1618
```

## PRIME-RL Training
Use this under your PRIME-RL config's `[[orchestrator.env]]` section.

```toml
[[orchestrator.env]]
id = "pubmedqa"
name = "pubmedqa"
args = { use_think = true, train_answer_formats = "random", training_shuffle_answers = true, training_seed = 23 }
```

Training notes:
- `training_shuffle_answers = true` reshuffles answer order at rollout time for train rows only.
- The stored training dataset remains stable; repeated RL presentations of the same row may use different answer orders.
- Eval rows remain fixed unless `shuffle_answers = true` is explicitly enabled.
- `training_seed` controls training answer-format routing. Rollout-time train reshuffling itself is stochastic.

### Model Input Format

Prompts are chat-formatted with a system prompt determined by `answer_format` and `use_think`. The user message contains a PubMedQA multiple-choice prompt in this shape:

The message contents look like:
```
Select the best answer.

Context: BACKGROUND. ...
RESULTS. ...

Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
A. Yes
B. No
C. Maybe
Answer:
```

For eval, choices remain fixed unless `shuffle_answers = true`. For RL training, `training_shuffle_answers = true` reshuffles the presented answer order at rollout time.



### Credits

For the original publication, cite:
```bibtex
@article{jin2019pubmedqa,
  title={Pubmedqa: A dataset for biomedical research question answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William W and Lu, Xinghua},
  journal={arXiv preprint arXiv:1909.06146},
  year={2019}
}
```

The evaluation code draws strongly on https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/pubmedqa for consistency across implementations. 

### Authors
This environment has been put together by:

Robert Scholz - ([@rscgh](https://github.com/rscgh))
