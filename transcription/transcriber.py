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
