from embeddings.embedder import embed_chunks
from vectorstore.vector_index import build_faiss_index, save_faiss_index, load_faiss_index
import os
import numpy as np

# Load chunks
with open("cache/chunks.txt", "r", encoding="utf-8") as f:
    raw_chunks = f.read().split("[Chunk ")[1:]
    chunks = [chunk.split("]", 1)[1].strip() for chunk in raw_chunks]

# Embed
embeddings = embed_chunks(chunks)

# Build index
index = build_faiss_index(embeddings)

# Save to disk
save_faiss_index("cache/faiss_index", index, chunks)
print("✅ FAISS index and chunks saved.")

# Load back (test)
loaded_index, loaded_chunks = load_faiss_index("cache/faiss_index")
print(f"📦 Loaded {len(loaded_chunks)} chunks from FAISS.")
