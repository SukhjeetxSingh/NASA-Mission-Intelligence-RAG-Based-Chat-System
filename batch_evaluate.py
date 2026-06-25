#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from statistics import mean

import rag_client
import llm_client
import ragas_evaluator


DEFAULT_TEST_FILE = "test_questions.json"


def load_test_questions(file_path: str):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("test_questions.json must contain a non-empty list")

    return data


def find_collection():
    backends = rag_client.discover_chroma_backends()
    if not backends:
        raise RuntimeError("No ChromaDB backends found. Run embedding_pipeline.py first.")

    first_key = next(iter(backends))
    backend = backends[first_key]

    collection, success, error = rag_client.initialize_rag_system(
        backend["directory"],
        backend["collection_name"]
    )
    if not success:
        raise RuntimeError(f"Failed to initialize RAG system: {error}")

    return collection, backend


def run_single_evaluation(collection, sample, api_key, model="gpt-3.5-turbo", n_results=3):
    question = sample.get("question", "").strip()
    mission = sample.get("mission", "").strip() if sample.get("mission") else None

    if not question:
        return {"error": "Empty question"}

    docs_result = rag_client.retrieve_documents(
        collection=collection,
        query=question,
        n_results=n_results,
        mission_filter=mission
    )

    if not docs_result or not docs_result.get("documents"):
        return {"error": "No documents retrieved"}

    documents = docs_result["documents"][0]
    metadatas = docs_result["metadatas"][0]

    context = rag_client.format_context(documents, metadatas)

    if not context.strip():
        return {"error": "Empty formatted context"}

    answer = llm_client.generate_response(
        openai_key=api_key,
        user_message=question,
        context=context,
        conversation_history=[],
        model=model
    )

    scores = ragas_evaluator.evaluate_response_quality(
        question=question,
        answer=answer,
        contexts=documents
    )

    result = {
        "id": sample.get("id"),
        "question": question,
        "category": sample.get("category", ""),
        "mission": mission or "",
        "answer": answer,
        "context_count": len(documents),
        "scores": scores
    }
    return result


def summarize_results(results):
    valid = [r for r in results if "error" not in r and isinstance(r.get("scores"), dict)]
    if not valid:
        return {"error": "No valid evaluation results"}

    metric_values = {}
    for row in valid:
        for metric, value in row["scores"].items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(metric, []).append(float(value))

    summary = {
        "num_samples": len(valid),
        "metrics": {}
    }
    for metric, values in metric_values.items():
        summary["metrics"][metric] = {
            "mean": round(mean(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4)
        }

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch evaluate NASA RAG project")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to test_questions.json")
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model")
    parser.add_argument("--n-results", type=int, default=3, help="Top-k docs to retrieve")
    args = parser.parse_args()

    if not args.openai_key:
        print("ERROR: OpenAI API key is required. Use --openai-key or set OPENAI_API_KEY.")
        sys.exit(1)

    try:
        test_questions = load_test_questions(args.test_file)
        collection, backend = find_collection()

        print(f"Using Chroma backend: {backend['display_name']}")
        print(f"Loaded {len(test_questions)} test questions from {args.test_file}")
        print("=" * 80)

        results = []
        for i, sample in enumerate(test_questions, start=1):
            print(f"\n[{i}/{len(test_questions)}] {sample.get('question', '')}")
            result = run_single_evaluation(
                collection=collection,
                sample=sample,
                api_key=args.openai_key,
                model=args.model,
                n_results=args.n_results
            )

            results.append(result)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Category: {result['category']}")
                print(f"  Mission: {result['mission']}")
                print(f"  Contexts Retrieved: {result['context_count']}")
                print(f"  Answer: {result['answer'][:250]}...")
                print("  Scores:")
                for metric, value in result["scores"].items():
                    if isinstance(value, (int, float)):
                        print(f"    {metric}: {value:.4f}")
                    else:
                        print(f"    {metric}: {value}")

        print("\n" + "=" * 80)
        summary = summarize_results(results)

        if "error" in summary:
            print(f"SUMMARY ERROR: {summary['error']}")
        else:
            print(f"Valid samples: {summary['num_samples']}")
            print("Aggregate metrics:")
            for metric, stats in summary["metrics"].items():
                print(
                    f"  {metric}: "
                    f"mean={stats['mean']:.4f}, "
                    f"min={stats['min']:.4f}, "
                    f"max={stats['max']:.4f}"
                )

    except Exception as e:
        print(f"Batch evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()