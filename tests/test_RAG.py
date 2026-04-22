# ====================== tests/test_RAG.py ======================

from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.embedding import get_embedding_function
from config.settings import CHROMA_PATH, LLM_MODEL, TOP_K

import matplotlib.pyplot as plt


# ====================== PROMPTS ======================
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

EVAL_PROMPT = """
You are an evaluator. Compare the actual answer with the expected answer.
Answer ONLY with 'true' or 'false'.
Expected answer: {expected_answer}
Actual answer: {actual_answer}
Answer with true or false only.
"""


# ====================== TEST: RETRIEVAL ======================
def test_retrieval(question: str, expected_ids: list, k: int = TOP_K):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    results = db.similarity_search_with_relevance_scores(question, k=k)
    retrieved_ids = [doc.metadata.get("id") for doc, _ in results]

    print(f"🔍 Question: {question}")
    print(f"✅ Expected IDs : {expected_ids}")
    print(f"Retrieved IDs  : {retrieved_ids}")

    hits = [eid for eid in expected_ids if eid in retrieved_ids]

    if hits:
        print(f"Result: ✅ PASS ({len(hits)} match(es))")
    else:
        print("Result: ❌ FAIL")

    print("-" * 80)
    return len(hits) > 0


# ====================== TEST: GENERATION ======================
def test_generation(context: str, question: str, expected_answer: str):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=question)

    model = OllamaLLM(model=LLM_MODEL, temperature=0)
    actual_answer = model.invoke(prompt)

    eval_prompt = EVAL_PROMPT.format(
        expected_answer=expected_answer,
        actual_answer=actual_answer,
    )

    judge = OllamaLLM(model=LLM_MODEL, temperature=0)
    judgment = judge.invoke(eval_prompt).strip().lower()

    result = judgment == "true"
    print(f"Generation Test → {'✅ PASS' if result else '❌ FAIL'}")
    return result


# ====================== HIT@K EVALUATION ======================
def evaluate_hit_at_k(test_cases, ks=[1, 3, 5]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    results_table = {k: 0 for k in ks}
    total = len(test_cases)

    for case in test_cases:
        question = case["question"]
        expected_ids = case["expected_ids"]

        results = db.similarity_search_with_relevance_scores(question, k=max(ks))
        retrieved_ids = [doc.metadata.get("id") for doc, _ in results]

        for k in ks:
            top_k_ids = retrieved_ids[:k]
            if any(eid in top_k_ids for eid in expected_ids):
                results_table[k] += 1

    # Print results
    print("\n📊 Hit@k Results:\n")
    for k in ks:
        print(f"Hit@{k}: {results_table[k] / total:.2%}")

    return results_table, total


# ====================== PLOT FUNCTION ======================
def plot_hit_at_k(results_table, total):
    import matplotlib.pyplot as plt

    ks = list(results_table.keys())
    hit_rates = [results_table[k] / total for k in ks]

    plt.figure()

    # Bar chart
    plt.bar(ks, hit_rates)

    plt.title("Hit@k Performance")
    plt.xlabel("k (Top-k Retrieved Chunks)")
    plt.ylabel("Hit Rate")

    plt.xticks(ks)
    plt.ylim(0, 1)

    # Add labels on top of bars
    for i, rate in enumerate(hit_rates):
        plt.text(ks[i], rate, f"{rate:.2%}", ha='center', va='bottom')

    plt.grid(axis='y')
    plt.savefig("hit_at_k.png", dpi=300, bbox_inches='tight')
    plt.show()

# ====================== TEST CASES ======================
test_cases = [
    {
        "question": "What is the definition of PRB Utilization and what is its warning threshold?",
        "expected_ids": [
            "data\\KPI_Thresholds.xlsx:2:0",
            "data\\3GPP_Technical_Specifications.pdf:37:0",
        ],
    },
    {
        "question": "What does 3GPP TS 28.554 define as the measurement period for handover KPIs?",
        "expected_ids": [
            "data\\3GPP_Technical_Specifications.pdf:73:0",
            "data\\3GPP_Technical_Specifications.pdf:41:0",
            "data\\3GPP_Technical_Specifications.pdf:42:0",
        ],
    },
    {
        "question": "What are the steps to follow when PRB utilization exceeds the critical threshold?",
        "expected_ids": [
            "data\\NOC_Runbook.txt:0:0",
            "data\\KPI_Thresholds.xlsx:2:0",
        ],
    },
    {
        "question": "Which network function is responsible for session management in 5G core?",
        "expected_ids": [
            "data\\3GPP_Technical_Specifications.pdf:65:2",
            "data\\3GPP_Technical_Specifications.pdf:13:1",
        ],
    },
    {
        "question": "What KPIs have a warning threshold below 90%?",
        "expected_ids": [
            "data\\KPI_Thresholds.xlsx:16:0",
            "data\\KPI_Thresholds.xlsx:14:0",
            "data\\KPI_Thresholds.xlsx:2:0",
            "data\\KPI_Thresholds.xlsx:5:0",
            "data\\KPI_Thresholds.xlsx:13:0",
            "data\\KPI_Thresholds.xlsx:6:0",
        ],
    },
]


# ====================== MAIN ======================
if __name__ == "__main__":
    print("=== RAG Evaluation Started ===\n")

    # Run individual retrieval tests
    for case in test_cases:
        test_retrieval(case["question"], case["expected_ids"])

    # Evaluate Hit@k
    results_table, total = evaluate_hit_at_k(test_cases)

    # Plot results
    plot_hit_at_k(results_table, total)

    print("\n🎉 Evaluation finished!")