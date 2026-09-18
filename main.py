"""
main.py
Runs the RAG pipeline over the evaluation question set, then scores
the results with RAGAS metrics:
  - faithfulness        (is the answer grounded in the retrieved context?)
  - answer_relevance    (does the answer address the question?)
  - context_precision   (are the retrieved chunks relevant?)
  - context_recall      (did retrieval find what was needed vs. ground truth?)

Outputs a per-question breakdown and an overall score summary,
and saves the full results to results.csv.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
# Ragas core imports
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper            
from ragas.embeddings import LangchainEmbeddingsWrapper  
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from rag_eval.eval_dataset import EVAL_QUESTIONS
from rag_eval.ragpipeline import RAGPipeline

from rag_eval.settings import CHAT_MODEL, EMBEDDING_MODEL


# START: Build the RAGAS dataset by running each evaluation question through the pipeline.
def build_eval_dataset(pipeline: RAGPipeline) -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in EVAL_QUESTIONS:
        print(f"Running pipeline on: {item['question']}")
        result = pipeline.query(item["question"])
        
        questions.append(item["question"])
        answers.append(result["answer"])
        
        # Ensure 'contexts' is always a list of strings: e.g., ["chunk 1", "chunk 2"]
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "user_input": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
# END: Build the RAGAS dataset by running each evaluation question through the pipeline.

raw_llm = ChatGroq(model=CHAT_MODEL, temperature=0)
raw_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Wrap them cleanly for Ragas compatibility
evaluator_llm = LangchainLLMWrapper(raw_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

# START: Run the RAG pipeline, evaluate its answers, and save the scored results.
def main():
    pipeline = RAGPipeline()
    dataset = build_eval_dataset(pipeline)

    print("\nRunning RAGAS evaluation...\n")
    single_worker_config = RunConfig(max_workers=1, timeout=60)
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,  # Evaluates Context Relevance
            context_recall,     # Evaluates if ground truth matches context
            faithfulness,       # Evaluates if answer is grounded in context
            answer_relevancy,   # Evaluates if answer matches the question
    ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=single_worker_config,
        raise_exceptions=True 
    )

    print("\n=== Global Average Scores ===")
    print(result)

    df = result.to_pandas()
    df.to_csv("results.csv", index=False)

    print("\nFull results saved to results.csv")
# END: Run the RAG pipeline, evaluate its answers, and save the scored results.


if __name__ == "__main__":
    main()