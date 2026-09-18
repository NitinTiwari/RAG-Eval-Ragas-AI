"""
ingest.py
Loads the .txt documents from ./data, splits them into chunks,
embeds them, and persists a FAISS vector store to ./vectorstore.

Run this once before ragpipeline.py or main.py.
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .settings import DATA_DIR, EMBEDDING_MODEL, VECTORSTORE_DIR


# START: Load, split, embed, and persist the source documents as a FAISS index.
def build_vectorstore():
    print(f"Loading documents from ./{DATA_DIR} ...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to ./{VECTORSTORE_DIR}")
# END: Load, split, embed, and persist the source documents as a FAISS index.


if __name__ == "__main__":
    build_vectorstore()
