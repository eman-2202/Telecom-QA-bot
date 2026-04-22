# ====================== config/real_id.py ======================
# Utility: list all chunk IDs currently stored in ChromaDB

from langchain_chroma import Chroma
from src.embedding import get_embedding_function
from config.settings import CHROMA_PATH

db  = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function(),
    collection_metadata={"hnsw:space": "cosine"},
)
ids = db.get(include=[])["ids"]

print("\n=== YOUR REAL CHUNK IDs ===\n")
for i, cid in enumerate(ids):
    print(f"{i+1:2d}. {cid}")
print(f"\nTotal chunks = {len(ids)}")