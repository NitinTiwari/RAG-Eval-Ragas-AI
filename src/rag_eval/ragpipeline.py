"""
ragpipeline.py
Core RAG pipeline: retrieve relevant chunks from the FAISS store,
then generate a grounded answer with the chat model.

Can be run directly for a quick interactive query, or imported
by main.py to generate answers for the RAGAS evaluation set.
"""

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from .settings import CHAT_MODEL, EMBEDDING_MODEL, VECTORSTORE_DIR

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
context provided below. If the answer is not contained in the context,
say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


# START: Manage retrieval, grounded answer generation, and complete RAG queries.
class RAGPipeline:
    # START: Load the configured embedding model, vector store, chat model, and prompt.
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
        )
        self.llm = ChatGroq(model=CHAT_MODEL, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # END: Load the configured embedding model, vector store, chat model, and prompt.

    # START: Retrieve the most relevant document chunks for a question.
    def retrieve(self, question: str):
        docs = self.vectorstore.similarity_search(question, k=self.top_k)
        return [d.page_content for d in docs]
    # END: Retrieve the most relevant document chunks for a question.

    # START: Generate a grounded answer from the supplied question and contexts.
    def generate(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(contexts)
        messages = self.prompt.format_messages(context=context_text, question=question)
        response = self.llm.invoke(messages)
        return response.content
    # END: Generate a grounded answer from the supplied question and contexts.

    # START: Retrieve context and generate an answer for one user question.
    def query(self, question: str):
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return {"question": question, "answer": answer, "contexts": contexts}
    # END: Retrieve context and generate an answer for one user question.
# END: Manage retrieval, grounded answer generation, and complete RAG queries.


if __name__ == "__main__":
    pipeline = RAGPipeline()
    print("RAG pipeline ready. Type a question (or 'exit' to quit).\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        result = pipeline.query(q)
        print(f"\nAnswer: {result['answer']}\n")
        print("Retrieved context snippets:")
        for i, c in enumerate(result["contexts"], 1):
            print(f"  [{i}] {c[:150]}...")
        print()
