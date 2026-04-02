# 🎬 YouTube RAG Q\&A Assistant

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
git clone https://github.com/Gtest111/Ragna-Vid
cd youtube-rag-qa

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
  [mistral-7b-instruct-v0.1.Q4\_K\_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
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
