from transcription.transcriber import transcribe_audio

audio_file = "cache/audio/audio.mp3"
transcript = transcribe_audio(audio_file)

print("\n--- Transcript Preview ---\n")
print(transcript[:1000] + "...\n")  # show first 1000 characters
