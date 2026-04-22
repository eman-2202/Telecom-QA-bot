# ====================== src/embedding.py ======================
from langchain_ollama import OllamaEmbeddings
from config.settings import EMBEDDING_MODEL_NAME


def get_embedding_function():
    """Returns the bge-m3 embedding model (exactly as taught in the course)"""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    return embeddings