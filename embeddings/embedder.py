# from sentence_transformers import SentenceTransformer
# import numpy as np

# # Load once
# model = SentenceTransformer("thenlper/gte-small")

# def embed_chunks(chunks: list[str]) -> np.ndarray:
#     """
#     Returns a NumPy array of embeddings for a list of text chunks.
#     """
#     embeddings = model.encode(chunks, show_progress_bar=True)
#     return np.array(embeddings)

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once — use quantized model if available for speed
model = SentenceTransformer("thenlper/gte-small")

def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Generate high-quality sentence embeddings for the provided chunks.
    Applies normalization for better FAISS indexing.
    """
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Recommended for cosine similarity
    )
    return embeddings
