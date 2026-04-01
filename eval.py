import json
import os
import time
import argparse
from typing import Iterable, List, Set

from embeddingRAG import SystemConfig, build_system


def normalize_source(path: str) -> str:
    return os.path.basename(path)


def normalize_sources(paths: Iterable[str]) -> Set[str]:
    return {normalize_source(p) for p in paths if p}


def keyword_match_score(answer: str, keywords: List[str]) -> float:
    answer = (answer or "").lower()
    hits = sum(1 for k in keywords if (k or "").lower() in answer)
    return hits / len(keywords) if keywords else 0.0


def source_match_score(pred_sources: List[str], true_sources: List[str]) -> float:
    pred = normalize_sources(pred_sources)
    true = normalize_sources(true_sources)
    return len(pred & true) / len(true) if true else 0.0


def selected_chunk_sources(result) -> List[str]:
    chunks = getattr(result, "_selected_chunks", None) or []
    return [ch.source_path for ch in chunks if getattr(ch, "source_path", None)]


def retrieval_metrics(result, expected_sources: List[str]):
    expected = normalize_sources(expected_sources)
    selected_sources = normalize_sources(selected_chunk_sources(result))

    if not expected:
        return {
            "retrieval_hit": 0.0,
            "retrieval_recall": 0.0,
            "selected_source_score": 0.0,
            "selected_sources": sorted(selected_sources),
        }

    overlap = selected_sources & expected
    retrieval_hit = 1.0 if overlap else 0.0
    retrieval_recall = len(overlap) / len(expected)
    selected_source_score = len(overlap) / len(expected)

    return {
        "retrieval_hit": retrieval_hit,
        "retrieval_recall": retrieval_recall,
        "selected_source_score": selected_source_score,
        "selected_sources": sorted(selected_sources),
    }


def evaluate(
    dataset_path: str = "eval_set.json",
    data_dir: str = "data",
    memory_store_dir: str = ".conv_memory_eval",
    ollama_model: str = "llama3:8b",
    debug: bool = False,
    mode: str = "full",
    hybrid_alpha: float = 0.6,
):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = SystemConfig(
        data_dir=data_dir,
        ollama_model=ollama_model,
        debug=debug,
        memory_store_dir=memory_store_dir,
        mode=mode,
        hybrid_alpha=hybrid_alpha,
    )
    system, config = build_system(config)

    results = []
    for item in dataset:
        query = item["query"]
        expected_keywords = item["expected_answer_keywords"]
        expected_sources = item["relevant_sources"]

        t0 = time.time()
        result, dbg = system.answer(
            user_query=query,
            top_k=config.top_k,
            min_score=config.min_score,
            ollama_model=config.ollama_model,
        )
        latency_sec = time.time() - t0

        acc = keyword_match_score(result.answer, expected_keywords)
        src = source_match_score(result.used_sources, expected_sources)
        ret = retrieval_metrics(result, expected_sources)

        results.append({
            "query": query,
            "mode": mode,
            "answer": result.answer,
            "used_sources": result.used_sources,
            "selected_sources": ret["selected_sources"],
            "accuracy": acc,
            "source_score": src,
            "retrieval_hit": ret["retrieval_hit"],
            "retrieval_recall": ret["retrieval_recall"],
            "selected_source_score": ret["selected_source_score"],
            "latency_sec": round(latency_sec, 4),
            "top_scores": result.top_scores,
            "had_low_confidence": result.had_low_confidence,
            "judge_rejected": result.judge_rejected,
            "debug": dbg,
        })

    return results


def summarize(results):
    avg_acc = sum(r["accuracy"] for r in results) / len(results)
    avg_src = sum(r["source_score"] for r in results) / len(results)
    avg_ret_hit = sum(r["retrieval_hit"] for r in results) / len(results)
    avg_ret_recall = sum(r["retrieval_recall"] for r in results) / len(results)
    avg_sel_src = sum(r["selected_source_score"] for r in results) / len(results)
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)

    print("Average Accuracy:", round(avg_acc, 3))
    print("Source Attribution:", round(avg_src, 3))
    print("Retrieval Hit Rate:", round(avg_ret_hit, 3))
    print("Retrieval Recall:", round(avg_ret_recall, 3))
    print("Selected Source Score:", round(avg_sel_src, 3))
    print("Average Latency (s):", round(avg_latency, 3))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AgenticRAG baseline modes.")
    parser.add_argument("--dataset_path", type=str, default="eval_set.json")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--memory_store_dir", type=str, default=".conv_memory_eval")
    parser.add_argument("--ollama_model", type=str, default="llama3:8b")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["dense_only", "lexical_only", "single_agent", "full"],
    )
    parser.add_argument("--hybrid_alpha", type=float, default=0.6)
    parser.add_argument("--output_path", type=str, default="eval_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate(
        dataset_path=args.dataset_path,
        data_dir=args.data_dir,
        memory_store_dir=args.memory_store_dir,
        ollama_model=args.ollama_model,
        debug=args.debug,
        mode=args.mode,
        hybrid_alpha=args.hybrid_alpha,
    )
    summarize(results)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {args.output_path}")