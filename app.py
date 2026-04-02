# import streamlit as st
# from rag_pipeline import run_pipeline  # ✅ Use your full pipeline!
# import os

# st.set_page_config(page_title="RAG from YouTube", layout="centered")
# st.title("🎥 RAG from YouTube")
# st.markdown("Upload a YouTube video, transcribe it, and ask questions using AI!")

# # Step 1: YouTube URL Input
# url = st.text_input("📎 Paste YouTube video URL:")

# # Step 2: Question Input
# question = st.text_input("💬 Ask a question about the video")

# if st.button("🚀 Run Pipeline"):
#     if url and question:
#         with st.spinner("Processing video and answering your question..."):
#             answer = run_pipeline(url, question)
#         st.markdown("### 🤖 Answer")
#         st.success(answer)
#     else:
#         st.warning("Please provide both a YouTube URL and a question.")

# # Footer
# st.markdown("---")
# st.caption("Built with 💡 using Streamlit + FAISS + Local LLM")



# import streamlit as st
# from rag_pipeline import run_pipeline, get_hash_from_url

# # Set Streamlit page config
# st.set_page_config(page_title="🎥 Chat with YouTube", layout="centered")
# st.title("💬 Chat with YouTube Video")
# st.caption("Powered by RAG + Local LLM")

# # Initialize session state
# if "video_hash" not in st.session_state:
#     st.session_state.video_hash = None
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Input YouTube URL
# youtube_url = st.text_input("📺 Paste YouTube video URL")

# if youtube_url:
#     # Compute hash of this video
#     current_hash = get_hash_from_url(youtube_url)

#     # If this is a new video (not same as before), reset chat history
#     if st.session_state.video_hash != current_hash:
#         st.session_state.video_hash = current_hash
#         st.session_state.history = []
#         st.success("✅ Video loaded! You can now ask questions.")

#     # Chat UI
#     st.markdown("---")
#     st.subheader("💬 Ask questions about the video")

#     # Show chat history
#     for pair in st.session_state.history:
#         with st.chat_message("user"):
#             st.markdown(pair["question"])
#         with st.chat_message("assistant"):
#             st.markdown(pair["answer"])

#     # Accept input from user
#     question = st.chat_input("Type your question...")
#     if question:
#         with st.chat_message("user"):
#             st.markdown(question)

#         with st.spinner("Thinking..."):
#             answer = run_pipeline(youtube_url, question)

#         with st.chat_message("assistant"):
#             st.markdown(answer)

#         # Add to history
#         st.session_state.history.append({"question": question, "answer": answer})


import streamlit as st
from rag_pipeline import run_pipeline, get_hash_from_url, get_transcript_path
import requests
import re
import os

# Streamlit Page Settings
st.set_page_config(page_title="Chat with YouTube", layout="centered")
st.title("💬 Chat with YouTube Video")

# Session State Init
if "video_hash" not in st.session_state:
    st.session_state.video_hash = None
if "history" not in st.session_state:
    st.session_state.history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "llm_config" not in st.session_state:
    st.session_state.llm_config = {
        "temperature": 0.3,
        "model_path": "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    }

# Sidebar: Model Settings
with st.sidebar:
    st.header("🛠️ Model Settings")
    st.session_state.llm_config["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    st.session_state.llm_config["model_path"] = st.selectbox(
        "Choose LLM model",
        options=[
            "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
            "models/llama2-7b-chat.Q4_K_M.gguf"
        ]
    )

# Input URL
youtube_url = st.text_input("📺 Paste YouTube video URL")

def extract_video_id(url: str):
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return match.group(1) if match else None

if youtube_url:
    video_id = extract_video_id(youtube_url)
    current_hash = get_hash_from_url(youtube_url)

    if st.session_state.video_hash != current_hash:
        st.session_state.video_hash = current_hash
        st.session_state.history = []

        # Show thumbnail
        if video_id:
            st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", caption="YouTube Video Preview")

        st.success("✅ Video loaded! You can now chat about it.")

    st.markdown("---")
    st.subheader("💬 Ask your question")

    # Display history
    for pair in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(pair["question"])
        with st.chat_message("assistant"):
            st.markdown(pair["answer"])

    # Chat input
    question = st.chat_input("Type your question...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            answer, transcript = run_pipeline(
                youtube_url, question,
                model_path=st.session_state.llm_config["model_path"],
                temperature=st.session_state.llm_config["temperature"]
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.history.append({"question": question, "answer": answer})
        st.session_state.transcript = transcript

    # Show download transcript button
    if st.session_state.transcript:
        st.download_button(
            label="📄 Download Transcript",
            data=st.session_state.transcript,
            file_name="transcript.txt",
            mime="text/plain"
        )
