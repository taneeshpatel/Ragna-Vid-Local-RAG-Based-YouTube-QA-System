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
