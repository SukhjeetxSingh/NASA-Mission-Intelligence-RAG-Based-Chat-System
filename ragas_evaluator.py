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
    ok, cleaned_or_error = _validate_inputs(question, answer, contexts)
    if not ok:
        return {"error": cleaned_or_error}

    cleaned = cleaned_or_error

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics._answer_relevance import ResponseRelevancy
        from ragas.metrics._faithfulness import Faithfulness
        from ragas import evaluate
    except Exception as e:
        return {"error": f"RAGAS import failed: {e}"}

    try:
        llm = llm_factory("gpt-3.5-turbo", client=OpenAI())
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=cleaned
        )

        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            ResponseRelevancy(llm=llm, embeddings=emb),
            Faithfulness(llm=llm),
        ]

        result = evaluate(dataset, metrics=metrics, show_progress=False, raise_exceptions=True)

        df = result.to_pandas()
        if df.empty:
            return {"error": "Evaluation produced no scores"}

        row = df.iloc[0].to_dict()
        output = {}
        for k, v in row.items():
            if k in ("user_input", "response", "retrieved_contexts", "reference"):
                continue
            try:
                output[str(k)] = float(v)
            except Exception:
                pass

        # --- ROUGE-L (additional metric, no API calls needed) ---
        try:
            from rouge_score import rouge_scorer as rs
            scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
            combined_context = " ".join(cleaned)
            rouge = scorer.score(combined_context, answer)
            output["rouge_l"] = round(rouge["rougeL"].fmeasure, 4)
        except ImportError:
            output["rouge_l_error"] = "rouge-score not installed. Run: pip install rouge-score"
        except Exception as e:
            output["rouge_l_error"] = f"ROUGE failed: {e}"

        return output or {"error": "No numeric scores returned"}

    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}