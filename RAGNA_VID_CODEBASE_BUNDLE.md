# Ragna-Vid Complete Codebase Bundle

This document bundles the complete, unabridged source code and configuration files for the **Ragna-Vid** project (Local YouTube Retrieval-Augmented Generation Q&A Assistant). It is structured for code review, system design analysis, and technical interview preparation.

---

## 📑 Architecture Overview & Table of Contents

### System Pipeline
1. **Audio Ingestion**: Downloads audio tracks from YouTube URLs with `yt-dlp` and `ffmpeg` (with automatic Windows binary discovery).
2. **Speech-to-Text**: Transcribes audio to text via `faster-whisper` (CTranslate2-backed Whisper) with VAD filtering and GPU/CPU fallback.
3. **Chunking**: Sentence-boundary-aware text chunking with word limit thresholds (~200 words).
4. **Embedding Generation**: Generates 384-dimensional normalized dense vectors with `thenlper/gte-small` using `sentence-transformers`.
5. **Vector Indexing & Persistence**: FAISS L2 indexing with disk persistence (`.index` + chunk pickle file).
6. **Retriever & QA Inference**: Cosine-similarity chunk retrieval coupled with quantized local LLM generation using `llama-cpp-python` (Mistral-7B-Instruct GGUF) supporting token streaming.
7. **Frontend Application**: Interactive Streamlit UI featuring parameter tuning (temperature, top-k), chat history, live transcript download, and video thumbnail previews.

### Table of Contents
- [Configuration & Environment](#configuration--environment)
  - [`.gitignore`](#--file-gitignore--)
  - [`requirements.txt`](#--file-requirementstxt--)
  - [`README.md`](#--file-readmemd--)
- [Core Application & Orchestration](#core-application--orchestration)
  - [`app.py`](#--file-apppy--)
  - [`rag_pipeline.py`](#--file-rag_pipelinepy--)
  - [`yt_qna_agent.py`](#--file-yt_qna_agentpy--)
- [Pipeline Modules](#pipeline-modules)
  - [`downloader/audio_downloader.py`](#--file-downloaderaudio_downloaderpy--)
  - [`transcription/transcriber.py`](#--file-transcriptiontranscriberpy--)
  - [`chunking/chunker.py`](#--file-chunkingchunkerpy--)
  - [`embeddings/embedder.py`](#--file-embeddingsembedderpy--)
  - [`vectorstore/vector_index.py`](#--file-vectorstorevector_indexpy--)
  - [`rag/retriever.py`](#--file-ragretrieverpy--)
  - [`rag/qa_engine.py`](#--file-ragqa_enginepy--)
- [Test & Verification Scripts](#test--verification-scripts)
  - [`testDownload.py`](#--file-testdownloadpy--)
  - [`test_transcribe.py`](#--file-test_transcribepy--)
  - [`test_chunk.py`](#--file-test_chunkpy--)
  - [`test_embed.py`](#--file-test_embedpy--)
  - [`test_faiss.py`](#--file-test_faisspy--)
  - [`test_rag.py`](#--file-test_ragpy--)

---

## Configuration & Environment

### --- FILE: .gitignore ---
```gitignore
venv
.vscode
__pycache__
```

---

### --- FILE: requirements.txt ---
```text
# Project dependencies for Ragna-Vid
streamlit>=1.38.0
yt-dlp>=2024.8.6
faster-whisper>=1.0.3
sentence-transformers>=3.0.1
faiss-cpu>=1.9.0
requests>=2.32.0
torchvision>=0.19.0
nvidia-cublas-cu12>=12.0
nvidia-cuda-runtime-cu12>=12.0
nvidia-cuda-nvrtc-cu12>=12.0
https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90-cu124/llama_cpp_python-0.2.90-cp312-cp312-win_amd64.whl
```

---

### --- FILE: README.md ---
```markdown
# 🎬 YouTube RAG Q&A Assistant

A powerful local Retrieval-Augmented Generation (RAG) system that turns any YouTube video into a knowledge base. Paste a video URL, transcribe its content, generate semantic embeddings, and query it with natural language — all locally with no API keys required.

Built using:

* `faster-whisper` for fast, accurate speech-to-text
* `thenlper/gte-small` for embedding transcript chunks
* FAISS for semantic search
* `llama-cpp-python` with Mistral-7B-Instruct GGUF for local LLM answers
* Streamlit for a clean and interactive frontend

---

## ✨ Features

* 🔗 Paste any YouTube video link
* 🔊 Download and transcribe audio using Whisper (faster-whisper)
* 📚 Chunk and embed transcripts with sentence-transformers
* 🔍 Retrieve relevant transcript sections using FAISS
* 🤖 Answer questions with a local quantized LLM (Mistral-7B)
* 🎛️ Customizable temperature for creativity
* 📥 Download transcript
* 🎞️ Watch video thumbnail preview

---

## 🛠️ Requirements

* ✅ Python 3.10+
* ✅ [FFmpeg](https://ffmpeg.org/download.html) – make sure it's added to your system PATH
* ✅ `llama-cpp-python` – for running local LLMs with GPU/CPU (cuBLAS preferred for GPU)
* ✅ Quantized GGUF model – e.g., `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
* ✅ Whisper model weights – auto-downloaded on first run (uses `faster-whisper`)
* ✅ Git LFS – required if cloning GGUF models from Hugging Face or GitHub

---

## 📦 Installation

```bash
git clone https://github.com/taneeshpatel/Ragna-Vid-Local-RAG-Based-YouTube-QA-System.git
cd Ragna-Vid-Local-RAG-Based-YouTube-QA-System

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate         # On Windows
# OR
source venv/bin/activate      # On macOS/Linux

# Install Python dependencies
pip install -r requirements.txt
```

---

## 🔧 Download Required Models

### 1. GGUF Model (Mistral)

* Download from Hugging Face or TheBloke:
  [mistral-7b-instruct-v0.1.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
* Place the `.gguf` file in the `models/` directory:

```bash
models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 2. Whisper

* No manual setup required. The `faster-whisper` model will be downloaded automatically (default: `base`).

### 3. Embedding Model

* Uses [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) from Hugging Face
* Automatically fetched via `sentence-transformers`

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
youtube-rag-qa/
├── app.py                      # Streamlit UI
├── rag_pipeline.py            # End-to-end RAG pipeline
├── downloader/
│   └── audio_downloader.py
├── transcription/
│   └── transcriber.py
├── chunking/
│   └── chunker.py
├── embeddings/
│   └── embedder.py
├── vectorstore/
│   └── vector_index.py
├── rag/
│   ├── retriever.py
│   └── qa_engine.py
├── models/                    # Place GGUF model here
├── cache/                     # Stores audio, transcript, chunks, FAISS index
└── requirements.txt
```
 
 ## 🧪 How It Works
```
Download Audio: YouTube URL is processed and audio is downloaded using yt-dlp and ffmpeg.

Transcription: Audio is transcribed locally using faster-whisper.

Chunking: Text is broken into ~200-word chunks based on sentence boundaries.

Embedding: Chunks are embedded with GTE-small.

Indexing: FAISS is used to create a fast similarity index.

RAG QA: A question is embedded and the most relevant chunks are retrieved. Then, Mistral 7B (GGUF) answers the question with the provided context.

```

  ## 📸 Screenshots

🎯 Step 1: Insert YouTube Link, Select Temperature, and Preview Video

![image](https://github.com/user-attachments/assets/84c77345-3151-46ea-be13-72fdf6275249)

💬 Step 2: Ask Your Question and Get an AI-Powered Answer

![image](https://github.com/user-attachments/assets/5df98743-1bc7-4aff-918c-d7b8bcb494a1)

---

## 📌 Future Improvements

* ✅ Add chat-like memory support (multi-turn QA)
* ✅ Live transcript editor for chunk refining
* ✅ Model selector with temperature, top-k toggles
* ✅ Switch between multiple GGUF models (optional)

---

## 💡 Credits

* [Mistral-7B-Instruct GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
* [faster-whisper](https://github.com/guillaumekln/faster-whisper)
* [FAISS](https://github.com/facebookresearch/faiss)
* [sentence-transformers](https://www.sbert.net/)
* [Streamlit](https://streamlit.io/)

---

## 📜 License

This project is licensed under MIT. Use responsibly.

---

Made with 💙 by Yajuvendrasinh Chudasama
```

---

## Core Application & Orchestration

### --- FILE: app.py ---
```python
import os
import re
import streamlit as st
from rag_pipeline import (
    prepare_video,
    query_video_stream,
    get_hash_from_url,
    get_transcript_path
)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Ragna-Vid | Local YouTube RAG",
    page_icon="🎬",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "video_hash" not in st.session_state:
    st.session_state.video_hash = None
if "history" not in st.session_state:
    st.session_state.history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "current_url" not in st.session_state:
    st.session_state.current_url = ""

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Settings")

    # Discover available GGUF models dynamically
    models_dir = "models"
    available_models = []
    if os.path.exists(models_dir):
        available_models = [
            os.path.join(models_dir, f).replace("\\", "/")
            for f in os.listdir(models_dir)
            if f.endswith(".gguf")
        ]

    if not available_models:
        available_models = ["models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"]

    selected_model = st.selectbox(
        "LLM Model",
        options=available_models,
        index=0,
        help="GGUF models detected in models/ directory"
    )

    temperature = st.slider(
        "Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower values are more factual; higher values are more creative."
    )

    top_k = st.slider(
        "Retrieved Chunks (Top-K)",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of transcript sections provided to LLM."
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    if st.session_state.transcript:
        st.download_button(
            label="📄 Download Transcript",
            data=st.session_state.transcript,
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True
        )

# Main UI Header
st.markdown('<div class="main-title">🎬 Ragna-Vid: Chat with YouTube</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Local RAG Assistant powered by faster-whisper, GTE-small, FAISS & Mistral 7B</div>', unsafe_allow_html=True)

# Helper: Extract YouTube ID (including regular URLs, short URLs, and shorts)
def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", url)
    return match.group(1) if match else None

col1, col2 = st.columns([3, 1])
with col1:
    youtube_url = st.text_input(
        "Paste YouTube video URL:",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        value=st.session_state.current_url
    )

# Process Video
if youtube_url and youtube_url.strip():
    cleaned_url = youtube_url.strip()
    video_id = extract_video_id(cleaned_url)
    current_hash = get_hash_from_url(cleaned_url)

    with col2:
        if video_id:
            st.image(
                f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                caption="Video Thumbnail",
                use_container_width=True
            )

    # When URL changes or video not yet processed
    if st.session_state.video_hash != current_hash:
        st.session_state.current_url = cleaned_url
        st.session_state.video_hash = current_hash
        st.session_state.history = []
        st.session_state.transcript = ""

        with st.status("🎬 Processing Video (Audio -> Whisper -> FAISS)...", expanded=True) as status:
            def update_status(msg):
                status.write(msg)

            success, v_hash, transcript = prepare_video(cleaned_url, status_callback=update_status)

            if success:
                st.session_state.transcript = transcript
                status.update(label="✅ Video processed and ready for questions!", state="complete", expanded=False)
            else:
                status.update(label="❌ Failed to process video.", state="error", expanded=True)
                st.error("Could not download or transcribe the video. Check that the URL is public and valid.")

    # Chat Interface
    st.markdown("### 💬 Chat")

    # Render Chat History
    for pair in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(pair["question"])
        with st.chat_message("assistant"):
            st.markdown(pair["answer"])

    # Chat Input & Streaming Output
    question = st.chat_input("Ask anything about this video (summarize, explain, write code, etc.)...")
    if question:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            stream_gen = query_video_stream(
                question=question,
                video_hash=st.session_state.video_hash,
                model_path=selected_model,
                temperature=temperature,
                top_k=top_k
            )
            full_answer = st.write_stream(stream_gen)

        # Save to chat history
        st.session_state.history.append({"question": question, "answer": full_answer})
else:
    st.info("👆 Paste a YouTube link above to transcribe the video and start chatting.")
```

---

### --- FILE: rag_pipeline.py ---
```python
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
from rag.qa_engine import generate_answer, generate_answer_stream


def get_hash_from_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def get_transcript_path(youtube_url: str, cache_dir: str = "cache") -> str:
    video_hash = get_hash_from_url(youtube_url)
    return os.path.join(cache_dir, f"{video_hash}_transcript.txt")


def prepare_video(youtube_url: str, cache_dir: str = "cache", status_callback=None) -> tuple[bool, str, str]:
    """
    Ensures video audio is downloaded, transcribed, chunked, and indexed in FAISS.
    Returns (success, video_hash, transcript).
    """
    video_hash = get_hash_from_url(youtube_url)
    audio_path = os.path.join(cache_dir, "audio", f"{video_hash}.mp3")
    transcript_path = os.path.join(cache_dir, f"{video_hash}_transcript.txt")
    chunks_path = os.path.join(cache_dir, f"{video_hash}_chunks.txt")
    index_path = os.path.join(cache_dir, f"{video_hash}_faiss_index")

    # Step 1: Download audio
    if not os.path.exists(audio_path):
        if status_callback:
            status_callback("⬇️ Downloading video audio...")
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        audio_path = download_audio(youtube_url, cache_dir=os.path.join(cache_dir, "audio"))
        if not audio_path:
            return False, video_hash, ""

    # Step 2: Transcription
    if not os.path.exists(transcript_path):
        if status_callback:
            status_callback("🎙️ Transcribing audio with Whisper...")
        transcript = transcribe_audio(audio_path)
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    else:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()

    # Step 3: Chunking
    if not os.path.exists(chunks_path):
        if status_callback:
            status_callback("✂️ Chunking transcript...")
        chunks = chunk_text(transcript, max_words=200)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"[Chunk {i+1}]\n{chunk}\n\n")
    else:
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("[Chunk ")[1:]
            chunks = [chunk.split("]", 1)[1].strip() for chunk in raw_chunks if "]" in chunk]

    # Step 4: Embedding + FAISS Index
    if not (os.path.exists(index_path + ".index") and os.path.exists(index_path + "_chunks.pkl")):
        if status_callback:
            status_callback("🔍 Generating semantic embeddings and building FAISS index...")
        embeddings = embed_chunks(chunks)
        index = build_faiss_index(embeddings)
        save_faiss_index(index_path, index, chunks)

    return True, video_hash, transcript


def query_video_stream(
    question: str,
    video_hash: str,
    model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature: float = 0.3,
    top_k: int = 4,
    cache_dir: str = "cache"
):
    """Retrieves relevant chunks and streams LLM answer tokens."""
    index_path = os.path.join(cache_dir, f"{video_hash}_faiss_index")
    relevant_chunks = retrieve_relevant_chunks(question, top_k=top_k, index_path=index_path)
    context = "\n\n".join(relevant_chunks)
    return generate_answer_stream(context, question, model_path=model_path, temperature=temperature)


def run_pipeline(
    youtube_url: str,
    question: str,
    model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    temperature: float = 0.3,
    cache_dir: str = "cache"
) -> tuple[str, str]:
    success, video_hash, transcript = prepare_video(youtube_url, cache_dir=cache_dir)
    if not success:
        return "❌ Failed to process or download video audio.", ""

    index_path = os.path.join(cache_dir, f"{video_hash}_faiss_index")
    relevant_chunks = retrieve_relevant_chunks(question, top_k=4, index_path=index_path)
    context = "\n\n".join(relevant_chunks)
    answer = generate_answer(context, question, model_path=model_path, temperature=temperature)
    return answer, transcript
```

---

### --- FILE: yt_qna_agent.py ---
```python
import os
from llama_cpp import Llama
import whisper
import yt_dlp

# Load local LLaMA model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=20,
    f16_kv=True,
    verbose=False
)

# Step 1: Download audio
def download_audio(youtube_url, out_path="audio.mp3"):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return out_path

# Step 2: Transcribe
def transcribe(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Step 3: Ask a question
def ask_question(transcript, question):
    prompt = f"""[INST] You are a helpful AI assistant.
You have the following video transcript:

{transcript}

Answer this question based on the transcript: {question} [/INST]"""
    output = llm(prompt, max_tokens=256, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# Usage
if __name__ == "__main__":
    yt_url = input("📺 Enter YouTube video URL: ")
    question = input("❓ What would you like to ask about the video? ")

    print("⬇️ Downloading and transcribing...")
    audio_path = download_audio(yt_url)
    transcript = transcribe(audio_path)

    print("💬 Generating answer...")
    answer = ask_question(transcript, question)
    print("\n🧠 Answer:", answer)
# https://www.youtube.com/watch?v=ad79nYk2keg
```

---

## Pipeline Modules

### --- FILE: downloader/audio_downloader.py ---
```python
# import os
# from yt_dlp import YoutubeDL

# def download_audio(url: str, output_path: str = "cache/audio/audio.mp3") -> str:
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     ydl_opts = {
#         'format': 'bestaudio/best',
#         'outtmpl': output_path,
#         'ffmpeg_location': 'C:/ffmpeg-7.1.1-essentials_build/bin',  # 🛠️ Set here
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',
#             'preferredquality': '192',
#         }],
#         'quiet': True,
#         'noplaylist': True
#     }

#     try:
#         with YoutubeDL(ydl_opts) as ydl:
#             ydl.download([url])
#         return output_path
#     except Exception as e:
#         print(f"[Error] Failed to download audio: {e}")
#         return None


import os
import sys
import glob
import shutil
import hashlib
from yt_dlp import YoutubeDL

# Ensure utf-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def find_ffmpeg() -> str | None:
    """Finds the directory containing ffmpeg.exe if not in PATH."""
    if shutil.which("ffmpeg"):
        return None

    # Search WinGet package install directory
    winget_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.exists(winget_dir):
        matches = glob.glob(os.path.join(winget_dir, "**", "ffmpeg.exe"), recursive=True)
        if matches:
            return os.path.dirname(matches[0])

    # Check common hardcoded locations
    for candidate in [
        r"C:\ffmpeg\bin",
        r"C:\ffmpeg-7.1.1-essentials_build\bin",
    ]:
        if os.path.exists(os.path.join(candidate, "ffmpeg.exe")):
            return candidate

    return None


def download_audio(url: str, cache_dir: str = "cache/audio") -> str:
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique filename based on URL hash
    video_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    output_base = os.path.join(cache_dir, video_hash)
    output_path = f"{output_base}.mp3"

    # Remove existing stale audio
    if os.path.exists(output_path):
        try:
            print(f"[Downloader] Removing old cached audio: {output_path}")
            os.remove(output_path)
        except Exception:
            pass

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_base}.%(ext)s",  # will be converted to .mp3
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'noplaylist': True
    }

    ffmpeg_dir = find_ffmpeg()
    if ffmpeg_dir:
        ydl_opts['ffmpeg_location'] = ffmpeg_dir

    try:
        print(f"[Downloader] Downloading audio for: {url}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Validate saved file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
            print(f"[Downloader] Audio downloaded successfully: {output_path}")
            return output_path
        else:
            print(f"[Downloader] Download failed or file too small: {output_path}")
            return None

    except Exception as e:
        print(f"[Downloader] Exception during download: {e}")
        return None
```

---

### --- FILE: transcription/transcriber.py ---
```python
# from faster_whisper import WhisperModel
# import os

# def transcribe_audio(audio_path: str, model_size="base") -> str:
#     model = WhisperModel(model_size, compute_type="int8", device="cpu")
#     segments, _ = model.transcribe(audio_path)

#     transcript = ""
#     for segment in segments:
#         transcript += segment.text + " "
#     return transcript.strip()


from faster_whisper import WhisperModel
import os
import torch

_model_cache = {}

def load_model(model_size="base") -> WhisperModel:
    if model_size not in _model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Whisper] Loading '{model_size}' model on {device}...")
        _model_cache[model_size] = WhisperModel(model_size, compute_type="int8", device=device)
    return _model_cache[model_size]

def transcribe_audio(audio_path: str, model_size="base") -> str:
    model = load_model(model_size)
    segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    transcript = " ".join(segment.text for segment in segments)
    return transcript.strip()
```

---

### --- FILE: chunking/chunker.py ---
```python
# import re

# def split_into_sentences(text: str) -> list:
#     # Basic sentence splitter using punctuation
#     sentences = re.split(r'(?<=[.!?])\s+', text.strip())
#     return sentences

# def chunk_text(text: str, max_words: int = 200) -> list:
#     sentences = split_into_sentences(text)
#     chunks = []
#     current_chunk = []
#     current_len = 0

#     for sentence in sentences:
#         word_count = len(sentence.split())

#         if current_len + word_count > max_words:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [sentence]
#             current_len = word_count
#         else:
#             current_chunk.append(sentence)
#             current_len += word_count

#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

import re

def split_into_sentences(text: str) -> list[str]:
    """
    Splits the input text into sentences using punctuation marks.
    Handles edge cases like abbreviations (e.g., Dr., i.e.) minimally.
    """
    # Basic sentence splitter, can be replaced with `nltk.sent_tokenize()` or spacy for better results
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_words: int = 200) -> list[str]:
    """
    Groups sentences into chunks, each with a maximum number of words.
    Ensures semantic boundaries and avoids cutting off mid-sentence.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        if word_count > max_words:
            # If a single sentence exceeds max_words, split it
            for i in range(0, word_count, max_words):
                sub_chunk = " ".join(words[i:i + max_words])
                chunks.append(sub_chunk)
            continue

        if current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

---

### --- FILE: embeddings/embedder.py ---
```python
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
```

---

### --- FILE: vectorstore/vector_index.py ---
```python
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
```

---

### --- FILE: rag/retriever.py ---
```python
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
```

---

### --- FILE: rag/qa_engine.py ---
```python
# from llama_cpp import Llama

# # Load your local GGUF model
# # llm = Llama(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=2048)
# llm = Llama(
#     model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     n_ctx=2048,
#     n_threads=8,         # Use available CPU cores
#     n_gpu_layers=32,     # If you have GPU via cuBLAS/Metal
#     temperature=0.3,     # More deterministic
#     top_p=0.9,
#     top_k=40,
#     repeat_penalty=1.1   # Prevents repeating phrases
# )

# def generate_answer(context: str, question: str) -> str:
#     prompt = f"""
# You are a helpful assistant. Use the context below to answer the question.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     output = llm(prompt, stop=["\n\n", "User:"], echo=False)
#     return output["choices"][0]["text"].strip()


# from llama_cpp import Llama

# # Load your local GGUF model with performance-tuned settings
# llm = Llama(
#     model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#     n_ctx=2048,
#     n_threads=8,         # Match to your CPU core count
#     n_gpu_layers=32,     # Optimize based on GPU availability
#     temperature=0.3,     # Lower temp = more deterministic answers
#     top_p=0.9,
#     top_k=40,
#     repeat_penalty=1.1
# )

# # Improved answer generation with better prompting
# def generate_answer(context: str, question: str) -> str:
#     prompt = f"""You are a knowledgeable assistant helping a user understand a transcript of a YouTube video.

# Use only the context below to answer the question truthfully. 
# If the answer is not clearly present, say: "I'm not sure based on the transcript."

# ---
# Context:
# \"\"\"{context}\"\"\"

# Question:
# {question}

# Answer:"""

#     output = llm(prompt, stop=["\n\n", "User:", "Question:"], echo=False, max_tokens=1024)
#     return output["choices"][0]["text"].strip()








import os
import sys

# Ensure Windows loads CUDA 12 and llama_cpp shared libraries
if sys.platform == "win32":
    site_packages = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages")
    candidate_dirs = [
        os.path.join(site_packages, "llama_cpp", "lib"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
        os.path.join(site_packages, "nvidia", "cuda_nvrtc", "bin"),
        os.path.join(site_packages, "torch", "lib"),
    ]
    for d in candidate_dirs:
        abs_d = os.path.abspath(d)
        if os.path.exists(abs_d):
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(abs_d)
                except Exception:
                    pass
            os.environ["PATH"] = abs_d + os.pathsep + os.environ.get("PATH", "")

from llama_cpp import Llama

# Cache the loaded LLM instance per model path (independent of temperature)
_loaded_models = {}

def get_optimal_threads() -> int:
    cpus = os.cpu_count() or 8
    return min(16, max(4, cpus - 2))

def get_llm(model_path: str) -> Llama:
    if model_path not in _loaded_models:
        threads = get_optimal_threads()
        # Offload 33 layers into RTX 3060 VRAM if GPU offload is supported
        gpu_layers = 33 if getattr(Llama, "llama_supports_gpu_offload", lambda: True)() else 0
        print(f"[QA Engine] Loading LLM from {model_path} (gpu_layers={gpu_layers}, threads={threads})...")
        _loaded_models[model_path] = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=threads,
            n_gpu_layers=gpu_layers,
            verbose=False
        )
    return _loaded_models[model_path]


def build_prompt(context: str, question: str) -> str:
    return f"""[INST] You are an intelligent and expert AI assistant analyzing a YouTube video transcript.

Context from the video transcript:
\"\"\"{context}\"\"\"

User Request:
{question}

Instructions:
1. Answer the user's request using the concepts, algorithms, and explanations from the video transcript.
2. If the user asks for code (such as C++, Python, Java, etc.) or practical algorithms related to the problem explained in the video, write clean, complete, and well-explained code implementing the solution.
3. If the request is entirely unrelated to the video topic, state that the transcript does not cover it.
4. Be concise, structured, and helpful. [/INST]"""


def generate_answer(context: str, question: str, model_path: str, temperature: float = 0.3) -> str:
    llm = get_llm(model_path)
    prompt = build_prompt(context, question)
    output = llm(
        prompt,
        stop=["</s>", "[INST]"],
        echo=False,
        max_tokens=1024,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1
    )
    return output["choices"][0]["text"].strip()


def generate_answer_stream(context: str, question: str, model_path: str, temperature: float = 0.3):
    llm = get_llm(model_path)
    prompt = build_prompt(context, question)
    for chunk in llm(
        prompt,
        stop=["</s>", "[INST]"],
        echo=False,
        max_tokens=1024,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
        stream=True
    ):
        text = chunk["choices"][0]["text"]
        if text:
            yield text
```

---

## Test & Verification Scripts

### --- FILE: testDownload.py ---
```python
from downloader.audio_downloader import download_audio

url = input("Paste a YouTube video URL: ")
audio_file = download_audio(url)

if audio_file:
    print(f"✅ Audio saved to: {audio_file}")
else:
    print("❌ Audio download failed.")
```

---

### --- FILE: test_transcribe.py ---
```python
from transcription.transcriber import transcribe_audio

audio_file = "cache/audio/audio.mp3"
transcript = transcribe_audio(audio_file)

print("\n--- Transcript Preview ---\n")
print(transcript[:1000] + "...\n")  # show first 1000 characters
```

---

### --- FILE: test_chunk.py ---
```python
from transcription.transcriber import transcribe_audio
from chunking.chunker import chunk_text
import os

# Paths
audio_file = "cache/audio/audio.mp3"
transcript_path = "cache/transcript.txt"
chunks_path = "cache/chunks.txt"

# Ensure cache dir exists
os.makedirs("cache", exist_ok=True)

# Transcription
print("🔁 Transcribing...")
transcript = transcribe_audio(audio_file)

# Save transcript
with open(transcript_path, "w", encoding="utf-8") as f:
    f.write(transcript)
print(f"📝 Transcript saved to: {transcript_path}")

# Chunking
print("📚 Chunking...")
chunks = chunk_text(transcript, max_words=200)

# Save chunks
with open(chunks_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"[Chunk {i+1}]\n{chunk}\n\n")
print(f"📦 Chunks saved to: {chunks_path}")

# Preview
print(f"\n✅ Total chunks: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...\n")
```

---

### --- FILE: test_embed.py ---
```python
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
```

---

### --- FILE: test_faiss.py ---
```python
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
```

---

### --- FILE: test_rag.py ---
```python
from rag.retriever import retrieve_relevant_chunks
from rag.qa_engine import generate_answer

question = input("❓ Ask a question about the video: ")

# Step 1: Retrieve relevant transcript chunks
top_chunks = retrieve_relevant_chunks(question, top_k=3)
context = "\n\n".join(top_chunks)

# Step 2: Generate answer from LLM
answer = generate_answer(context, question)

print("\n🤖 Answer:\n", answer)
```
