from embeddings.embedder import embed_chunks
import os

# Load chunks
with open("cache/chunks.txt", "r", encoding="utf-8") as f:
    raw_chunks = f.read().split("[Chunk ")[1:]  # split on markers
    chunks = [chunk.split("]", 1)[1].strip() for chunk in raw_chunks]

print(f"📦 Total chunks loaded: {len(chunks)}")

# Embed
embeddings = embed_chunks(chunks)

print(f"✅ Embedding shape: {embeddings.shape}")  # Should be (num_chunks, 384) for GTE-small
