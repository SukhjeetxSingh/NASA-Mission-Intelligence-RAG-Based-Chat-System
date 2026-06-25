from typing import Dict, List, Any


def _validate_inputs(question: str, answer: str, contexts: List[str]):
    if not question or not str(question).strip():
        return False, "Question is empty or malformed"
    if not answer or not str(answer).strip():
        return False, "Answer is empty or malformed"
    if contexts is None or not isinstance(contexts, list):
        return False, "Contexts must be a list of strings"
    cleaned = [str(c).strip() for c in contexts if str(c).strip()]
    if not cleaned:
        return False, "Retrieved context is empty"
    return True, cleaned


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics."""
    ok, cleaned_or_error = _validate_inputs(question, answer, contexts)
    if not ok:
        return {"error": cleaned_or_error}

    cleaned = cleaned_or_error

    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import SingleTurnSample, evaluate
        from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
    except Exception as e:
        return {"error": f"RAGAS import failed: {e}"}

    try:
        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=cleaned
        )

        metrics = [
            ResponseRelevancy(llm=llm, embeddings=emb),
            Faithfulness(llm=llm),
            ContextPrecision(),
        ]

        result = evaluate(sample, metrics=metrics)

        try:
            df = result.to_pandas()
            if df.empty:
                return {"error": "Evaluation produced no scores"}
            row = df.iloc[0].to_dict()
            output = {}
            for k, v in row.items():
                if k == "sample":
                    continue
                try:
                    output[str(k)] = float(v)
                except Exception:
                    pass
            return output or {"error": "No numeric scores returned"}
        except Exception:
            return {"error": "Could not convert evaluation results to table"}

    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}


def evaluate_batch_from_file(file_path: str) -> Dict[str, Any]:
    import json
    from pathlib import Path

    p = Path(file_path)
    if not p.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            data = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split("||")]
                if len(parts) >= 3:
                    contexts = json.loads(parts[2]) if parts[2].startswith("[") else [parts[2]]
                    data.append({
                        "question": parts[0],
                        "answer": parts[1],
                        "contexts": contexts
                    })

        rows = []
        metric_values = {}

        for item in data:
            question = item.get("question", "")
            answer = item.get("answer", "")
            contexts = item.get("contexts", [])

            result = evaluate_response_quality(question, answer, contexts)
            row = {"question": question, **result}
            rows.append(row)

            for k, v in result.items():
                if isinstance(v, (int, float)):
                    metric_values.setdefault(k, []).append(float(v))

        summary = {
            metric: (sum(values) / len(values) if values else None)
            for metric, values in metric_values.items()
        }

        return {
            "results": rows,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}