# emRAG

**emRAG** is a multi-agent RAG (Retrieval-Augmented Generation) prototype built around **embeddings + TF‑IDF hybrid retrieval** and a **three-tier memory architecture**.

It automatically builds one agent per subfolder in a dataset directory and provides an interactive CLI for querying said agents.

---

## Installation

Used Llama 3 8B locally with Ollama, install Ollama from the official website

```bash
ollama pull llama3:8b
ollama run llama3:8b

git clone git@github.com:xqzyr/emRAG.git
cd emRAG

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install sentence-transformers scikit-learn numpy requests
```

## Usage

### Start the system

```bash
python embeddingRAG.py --data_dir data
```

### CLI Arguments

```bash
python embeddingRAG.py [OPTIONS]
```

| Argument             | Description                                             | Default            |
| -------------------- | ------------------------------------------------------- | ------------------ |
| `--data_dir`         | Path to dataset directory                               | `data`             |
| `--embed_model`      | SentenceTransformer model                               | `all-MiniLM-L6-v2` |
| `--ollama_model`     | LLM used via Ollama                                     | `llama3:8b`        |
| `--top_k`            | Number of retrieved chunks                              | `5`                |
| `--min_score`        | Retrieval score threshold                               | `0.10`             |
| `--retrieval_mode`   | Retrieval type (`dense_only`, `lexical_only`, `hybrid`) | `hybrid`           |
| `--hybrid_alpha`     | Dense vs lexical weighting                              | `0.6`              |
| `--memory_store_dir` | Path for long-term memory                               | `.conv_memory`     |
| `--debug`            | Enable verbose debug output                             | `False`            |


### Evaluation 

```bash
python eval.py [OPTIONS]
```

| Argument             | Description                                                        | Default             |
| -------------------- | ------------------------------------------------------------------ | ------------------- |
| `--dataset_path`     | Evaluation dataset                                                 | `eval_set.json`     |
| `--data_dir`         | Data directory                                                     | `data`              |
| `--memory_store_dir` | Memory store for eval                                              | `.conv_memory_eval` |
| `--ollama_model`     | LLM model                                                          | `llama3:8b`         |
| `--mode`             | System mode (`dense_only`, `lexical_only`, `single_agent`, `full`) | `full`              |
| `--hybrid_alpha`     | Hybrid weighting                                                   | `0.6`               |
| `--output_path`      | Output JSON file                                                   | `eval_results.json` |
| `--debug`            | Debug mode                                                         | `False`             |

Example: 
```bash
python eval.py --mode dense_only --output_path eval_dense_only.json
python eval.py --mode lexical_only --output_path eval_lexical_only.json
python eval.py --mode single_agent --output_path eval_single_agent.json
python eval.py --mode full --output_path eval_full.json
```


On startup, the system builds or loads cached indexes and then enters interactive mode.

## Interactive Commands

- **add <path>**  
  Add a new folder or file and build an index.

  - **describe <agent-name>**  
  Use this to get Agent description.

- **clear**  
  Clears recent messages and summary memory.  
  Cached indexes on disk are preserved.

- **reset**  
  Clears recent + summary memory and deletes long-term storage on disk.  
  Use this for a clean-slate experiment.

- **exit / quit**  
  Exit the program.

---

## Dataset Structure

```
data/
  recipes/
    soup.txt
    pasta.txt
  tech/
    rag.txt
    containers.txt
  animals/
    cats.txt
```

Each subfolder becomes an agent. Use plain `.txt` files. The repository also contains the "data" used during development.

## Key Features

- **Multi-agent setup**: each subfolder becomes an independent agent
- **Hybrid retrieval**:
  - Dense semantic search (SentenceTransformers)
  - Sparse lexical ranking (TF‑IDF)
- **Three-tier memory**:
  1. Recent messages
  2. Conversation summary
  3. Retrieval fallback (vector + TF‑IDF)
- **Persistent on-disk cache** (`.rag_cache`) for fast restarts
- **Interactive CLI** with commands to add, clear, and reset memory

## How It Works

1. **Agent discovery**  
   Every subfolder inside `--data_dir` is treated as a separate agent.

2. **Indexing**
   - Text files are chunked into paragraphs
   - Chunks are embedded using a SentenceTransformer
   - TF‑IDF vectors are built for lexical matching
   - Results are cached to disk

3. **Query flow**
   - Use recent context if sufficient
   - Otherwise consult a summary
   - Otherwise retrieve relevant chunks using hybrid retrieval


