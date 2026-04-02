# rag_pipeline.py

# import os
# from downloader.audio_downloader import download_audio
# from transcription.transcriber import transcribe_audio
# from chunking.chunker import chunk_text
# from embeddings.embedder import embed_chunks
# from vectorstore.vector_index import build_faiss_index, save_faiss_index, load_faiss_index
# from rag.retriever import retrieve_relevant_chunks
# from rag.qa_engine import generate_answer


# def run_pipeline(youtube_url: str, question: str, cache_dir: str = "cache") -> str:
#     os.makedirs(f"{cache_dir}/audio", exist_ok=True)

#     # Step 1: Download audio
#     audio_path = download_audio(youtube_url)
#     if not audio_path:
#         return "❌ Failed to download audio."

#     # Step 2: Transcribe
#     transcript = transcribe_audio(audio_path)
#     transcript_path = f"{cache_dir}/transcript.txt"
#     with open(transcript_path, "w", encoding="utf-8") as f:
#         f.write(transcript)

#     # Step 3: Chunking
#     chunks = chunk_text(transcript, max_words=200)
#     with open(f"{cache_dir}/chunks.txt", "w", encoding="utf-8") as f:
#         for i, chunk in enumerate(chunks):
#             f.write(f"[Chunk {i+1}]\n{chunk}\n\n")

#     # Step 4: Embedding
#     embeddings = embed_chunks(chunks)

#     # Step 5: Build + Save FAISS index
#     index = build_faiss_index(embeddings)
#     save_faiss_index(f"{cache_dir}/faiss_index", index, chunks)

#     # Step 6: Retrieve relevant chunks
#     relevant_chunks = retrieve_relevant_chunks(question, top_k=3)
#     context = "\n\n".join(relevant_chunks)

#     # Step 7: Generate answer using local LLM
#     answer = generate_answer(context, question)

#     return answer
# import os
# import hashlib
# from downloader.audio_downloader import download_audio
# from transcription.transcriber import transcribe_audio
# from chunking.chunker import chunk_text
# from embeddings.embedder import embed_chunks
# from vectorstore.vector_index import build_faiss_index, save_faiss_index, load_faiss_index
# from rag.retriever import retrieve_relevant_chunks
# from rag.qa_engine import generate_answer

# def get_hash_from_url(url: str) -> str:
#     return hashlib.md5(url.encode()).hexdigest()[:8]

# def run_pipeline(youtube_url: str, question: str, cache_dir: str = "cache") -> str:
#     video_hash = get_hash_from_url(youtube_url)
#     audio_path = os.path.join(cache_dir, "audio", f"{video_hash}.mp3")
#     transcript_path = os.path.join(cache_dir, f"{video_hash}_transcript.txt")
#     chunks_path = os.path.join(cache_dir, f"{video_hash}_chunks.txt")
#     index_path = os.path.join(cache_dir, f"{video_hash}_faiss_index")

#     # Step 1: Download audio if not exists
#     if not os.path.exists(audio_path):
#         os.makedirs(os.path.dirname(audio_path), exist_ok=True)
#         audio_path = download_audio(youtube_url)
#         if not audio_path:
#             return "❌ Failed to download audio."

#     # Step 2: Transcription
#     if not os.path.exists(transcript_path):
#         transcript = transcribe_audio(audio_path)
#         with open(transcript_path, "w", encoding="utf-8") as f:
#             f.write(transcript)
#     else:
#         with open(transcript_path, "r", encoding="utf-8") as f:
#             transcript = f.read()

#     # Step 3: Chunking
#     if not os.path.exists(chunks_path):
#         chunks = chunk_text(transcript, max_words=200)
#         with open(chunks_path, "w", encoding="utf-8") as f:
#             for i, chunk in enumerate(chunks):
#                 f.write(f"[Chunk {i+1}]\n{chunk}\n\n")
#     else:
#         with open(chunks_path, "r", encoding="utf-8") as f:
#             raw_chunks = f.read().split("[Chunk ")[1:]
#             chunks = [chunk.split("]", 1)[1].strip() for chunk in raw_chunks]

#     # Step 4: Embedding + FAISS
#     if not os.path.exists(index_path + ".index") or not os.path.exists(index_path + "_chunks.pkl"):
#         embeddings = embed_chunks(chunks)
#         index = build_faiss_index(embeddings)
#         save_faiss_index(index_path, index, chunks)

#     # Step 5: Retrieve + Answer
#     relevant_chunks = retrieve_relevant_chunks(question, top_k=3, index_path=index_path)
#     context = "\n\n".join(relevant_chunks)
#     answer = generate_answer(context, question)

#     return answer







import os
import hashlib
from downloader.audio_downloader import download_audio
from transcription.transcriber import transcribe_audio
from chunking.chunker import chunk_text
from embeddings.embedder import embed_chunks
from vectorstore.vector_index import build_faiss_index, save_faiss_index
from rag.retriever import retrieve_relevant_chunks
from rag.qa_engine import generate_answer


def get_hash_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def get_transcript_path(youtube_url: str, cache_dir: str = "cache") -> str:
    video_hash = get_hash_from_url(youtube_url)
    return os.path.join(cache_dir, f"{video_hash}_transcript.txt")


def run_pipeline(
    youtube_url: str,
    question: str,
    model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature: float = 0.3,
    cache_dir: str = "cache"
) -> tuple[str, str]:
    video_hash = get_hash_from_url(youtube_url)

    # Define file paths
    audio_path = os.path.join(cache_dir, "audio", f"{video_hash}.mp3")
    transcript_path = os.path.join(cache_dir, f"{video_hash}_transcript.txt")
    chunks_path = os.path.join(cache_dir, f"{video_hash}_chunks.txt")
    index_path = os.path.join(cache_dir, f"{video_hash}_faiss_index")

    # Step 1: Download audio if not exists
    if not os.path.exists(audio_path):
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        audio_path = download_audio(youtube_url)
        if not audio_path:
            return "❌ Failed to download audio.", ""

    # Step 2: Transcription
    if not os.path.exists(transcript_path):
        transcript = transcribe_audio(audio_path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    else:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

    # Step 3: Chunking
    if not os.path.exists(chunks_path):
        chunks = chunk_text(transcript, max_words=200)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"[Chunk {i+1}]\n{chunk}\n\n")
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("[Chunk ")[1:]
            chunks = [chunk.split("]", 1)[1].strip() for chunk in raw_chunks]

    # Step 4: Embedding + FAISS
    if not os.path.exists(index_path + ".index") or not os.path.exists(index_path + "_chunks.pkl"):
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        save_faiss_index(index_path, index, chunks)

    # Step 5: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, top_k=3, index_path=index_path)
    context = "\n\n".join(relevant_chunks)

    # Step 6: Answer generation
    answer = generate_answer(context, question, model_path=model_path, temperature=temperature)

    return answer, transcript
