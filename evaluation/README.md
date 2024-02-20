# Evaluation

This repository contains code for evaluating the performance of LLM models through both single-turn and multi-turn scenarios.

To set up the environment, you can install the required dependencies by running:
```bash
pip install -r evaluation/requirements.txt
```

## Single-turn Evaluation

For single-turn evaluation, the processes of inference, post-processing, and result aggregation are separated as follows:

1. Execute `bash evaluation/evaluate/scripts/01_gen_single.sh` to generate model results.
2. Perform post-processing on the model output by executing `bash evaluation/evaluate/scripts/02_sanitize_single.sh`.
3. Finally, compute evaluation metrics by executing `bash evaluation/evaluate/scripts/03_eval_single.sh`.

## Multi-turn Evaluation

### Multi-turn Evaluation with Execution Feedback

Evaluate the performance of the models with execution feedback using the provided scripts:

- For OpenCodeInterpreter:
  ```bash
  bash evaluation/evaluate/scripts/04_execution_feedback_multiround_OpenCodeInterpreter.sh
  ```

- For OpenAI's GPT Models:
  Before proceeding with evaluation, ensure to implement the `get_predict` function in `chat_with_gpt.py` to enable interaction with the GPT Models. Then, execute the following script:
  ```bash
  bash evaluation/evaluate/scripts/05_execution_feedback_multiround_gpt.sh
  ```

### Multi-turn Evaluation with GPT-4 Simulated Human Feedback

Execute either of the following scripts to evaluate the models with simulated human feedback:

- For OpenCodeInterpreter:
  ```bash
  bash evaluation/evaluate/scripts/06_human_feedback_multiround_OpenCodeInterpreter.sh
  ```

- For Oracle OpenCodeInterpreter:
  ```bash
  bash evaluation/evaluate/scripts/07_human_feedback_multiround_Oracle_OpenCodeInterpreter.sh
  ```

These scripts facilitate the multi-turn evaluation with simulated human feedback.

This evaluation code is based on [EvalPlus](https://github.com/evalplus/evalplus) and has been modified for specific purposes. We extend our gratitude to the contributors of EvalPlus for their foundational work.
