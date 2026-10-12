# from sentence_transformers import SentenceTransformer
# from vectorstore.vector_index import load_faiss_index
# import numpy as np

# # Load embedding model and FAISS index
# model = SentenceTransformer("thenlper/gte-small")
# index, chunks = load_faiss_index("cache/faiss_index")

# def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), top_k)
#     return [chunks[i] for i in indices[0]]


# from sentence_transformers import SentenceTransformer
# from vectorstore.vector_index import load_faiss_index
# import numpy as np

# _model = None
# _index = None
# _chunks = None

# def get_model():
#     global _model
#     if _model is None:
#         print("🧠 Loading embedding model...")
#         _model = SentenceTransformer("thenlper/gte-small")
#     return _model

# def get_index_and_chunks():
#     global _index, _chunks
#     if _index is None or _chunks is None:
#         print("📦 Loading FAISS index and chunks...")
#         _index, _chunks = load_faiss_index("cache/faiss_index")
#     return _index, _chunks

# def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list[str]:
#     model = get_model()
#     index, chunks = get_index_and_chunks()
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), top_k)
#     return [chunks[i] for i in indices[0]]











from sentence_transformers import SentenceTransformer
from vectorstore.vector_index import load_faiss_index
import numpy as np

_model = None
_cached_indexes = {}  # Cache multiple indexes if needed

def get_model():
    global _model
    if _model is None:
        print("[Retriever] Loading embedding model...")
        _model = SentenceTransformer("thenlper/gte-small")
    return _model

def get_index_and_chunks(index_path="cache/faiss_index"):
    global _cached_indexes
    if index_path not in _cached_indexes:
        print(f"[Retriever] Loading FAISS index and chunks from: {index_path}")
        _cached_indexes[index_path] = load_faiss_index(index_path)
    return _cached_indexes[index_path]

# def retrieve_relevant_chunks(query: str, top_k: int = 5, index_path: str = "cache/faiss_index") -> list[str]:
#     model = SentenceTransformer("thenlper/gte-small")
#     index, chunks = load_faiss_index(index_path)
#     query_embedding = model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), top_k)
#     return [chunks[i] for i in indices[0]]

def retrieve_relevant_chunks(query: str, top_k: int = 5, index_path: str = "cache/faiss_index") -> list[str]:
    model = get_model()
    index, chunks = get_index_and_chunks(index_path)
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
