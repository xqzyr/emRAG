# Evaluation

## Overview

The project includes a dedicated evaluation script, `eval.py`, for measuring retrieval and answer quality across multiple system configurations.

The evaluation framework is designed to compare four modes:

- `dense_only`
- `lexical_only`
- `single_agent`
- `full`

This makes it possible to run ablation-style comparisons between simpler retrieval baselines and the complete multi-agent hybrid system.

## Evaluation Dataset

The evaluation set is stored in `eval_set.json`. Each example contains:

- `query`
- `expected_answer_keywords`
- `relevant_sources`

Example structure:

```json
{
  "query": "What is a GPU used for?",
  "expected_answer_keywords": ["parallel", "graphics", "machine learning"],
  "relevant_sources": ["GPUs.txt"]
}
```

The current dataset includes queries across multiple domains such as animals, technology, food, and space.

## Evaluation Script

The main entry point is:

```bash
python eval.py --mode full --output_path eval_full.json
```

The script loads the dataset, builds the configured system, runs inference for each query, computes evaluation metrics, and saves the per-example results to a JSON file.

## Supported Evaluation Modes

### Dense Only

```bash
python eval.py --mode dense_only --output_path eval_dense_only.json
```

Uses embedding-based retrieval only.

### Lexical Only

```bash
python eval.py --mode lexical_only --output_path eval_lexical_only.json
```

Uses TF-IDF retrieval only.

### Single Agent

```bash
python eval.py --mode single_agent --output_path eval_single_agent.json
```

Builds one unified RAG index over the full corpus rather than separate agents.

### Full System

```bash
python eval.py --mode full --output_path eval_full.json
```

Runs the full multi-agent hybrid system with routing, memory, and hybrid retrieval.

## Available CLI Arguments

### `eval.py`

```bash
python eval.py [OPTIONS]
```

| Argument | Description | Default |
|---|---|---|
| `--dataset_path` | Path to evaluation dataset | `eval_set.json` |
| `--data_dir` | Root dataset directory | `data` |
| `--memory_store_dir` | Directory for long-term evaluation memory | `.conv_memory_eval` |
| `--ollama_model` | Ollama model used for generation | `llama3:8b` |
| `--debug` | Enable debug output | disabled |
| `--mode` | Evaluation mode | `full` |
| `--hybrid_alpha` | Weight of dense score in hybrid retrieval | `0.6` |
| `--output_path` | Output file for per-query results | `eval_results.json` |

## Metrics

The evaluation script reports both answer-level and retrieval-level metrics.

### 1. Accuracy

`keyword_match_score` measures whether the generated answer contains the expected answer keywords.

```text
accuracy = matched_keywords / total_expected_keywords
```

This is a lightweight lexical proxy for factual correctness.

### 2. Source Attribution

`source_match_score` compares the filenames in `used_sources` with the gold `relevant_sources`.

```text
source_score = overlap(predicted_sources, expected_sources) / total_expected_sources
```

This measures whether the final answer cites the correct document sources.

### 3. Retrieval Hit Rate

A binary metric indicating whether any selected retrieved chunk came from at least one expected source.

### 4. Retrieval Recall

Measures how many of the expected source files appear among the selected chunk sources.

```text
retrieval_recall = overlap(selected_sources, expected_sources) / total_expected_sources
```

### 5. Selected Source Score

Equivalent to source overlap computed specifically over selected retrieved chunks.

### 6. Latency

For each query, the script records end-to-end runtime in seconds.

## Output Format

The evaluation results are written as JSON. Each result entry includes:

- query
- mode
- answer
- used_sources
- selected_sources
- accuracy
- source_score
- retrieval_hit
- retrieval_recall
- selected_source_score
- latency_sec
- top_scores
- had_low_confidence
- judge_rejected
- debug

This makes it possible to inspect not just aggregate scores but also failure patterns per example.

## Example Workflow

Run all four modes:

```bash
python eval.py --mode dense_only --output_path eval_dense_only.json
python eval.py --mode lexical_only --output_path eval_lexical_only.json
python eval.py --mode single_agent --output_path eval_single_agent.json
python eval.py --mode full --output_path eval_full.json
```

Then compare the aggregate outputs printed to the console:

- Average Accuracy
- Source Attribution
- Retrieval Hit Rate
- Retrieval Recall
- Selected Source Score
- Average Latency

## Interpretation

These four modes support a simple ablation design:

- `dense_only` measures the effect of semantic retrieval alone
- `lexical_only` measures the effect of sparse keyword-based retrieval alone
- `single_agent` measures the impact of removing agent-based corpus partitioning
- `full` measures the combined system with hybrid retrieval and agent routing

This setup helps isolate the contribution of each architectural choice.

## Notes on Evaluation Behavior

Because emRAG uses grounded answer generation and verification, the system may abstain when evidence is weak or unsupported. This means evaluation results reflect not only retrieval quality but also the strictness of answer validation.

The evaluation pipeline is therefore best interpreted as a combined test of:

- retrieval quality
- routing quality
- grounding quality
- abstention behavior

## Summary

The evaluation framework provides a reproducible way to compare the full system against simpler baselines. It captures both answer content and retrieval correctness, making it useful for analysis, debugging, and reporting in an experimental or thesis setting.
