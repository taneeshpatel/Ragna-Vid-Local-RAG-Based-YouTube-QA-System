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