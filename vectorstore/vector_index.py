# import faiss
# import numpy as np
# import os
# import pickle

# def save_faiss_index(index_path: str, faiss_index, chunks):
#     faiss.write_index(faiss_index, f"{index_path}.index")
#     with open(f"{index_path}_chunks.pkl", "wb") as f:
#         pickle.dump(chunks, f)

# def load_faiss_index(index_path: str):
#     index = faiss.read_index(f"{index_path}.index")
#     with open(f"{index_path}_chunks.pkl", "rb") as f:
#         chunks = pickle.load(f)
#     return index, chunks

# def build_faiss_index(embeddings: np.ndarray):
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index


import faiss
import numpy as np
import os
import pickle

def save_faiss_index(index_path: str, faiss_index, chunks: list[str]) -> None:
    """
    Saves the FAISS index and the associated chunks to disk.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(faiss_index, f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"[FAISS] Saved index to {index_path}.index and chunks to {index_path}_chunks.pkl")

def load_faiss_index(index_path: str):
    """
    Loads the FAISS index and associated chunks from disk.
    """
    index = faiss.read_index(f"{index_path}.index")
    with open(f"{index_path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def build_faiss_index(embeddings: np.ndarray):
    """
    Builds a FAISS L2 (Euclidean) index from the given embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index
