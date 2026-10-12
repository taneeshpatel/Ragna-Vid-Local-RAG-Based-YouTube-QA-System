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

