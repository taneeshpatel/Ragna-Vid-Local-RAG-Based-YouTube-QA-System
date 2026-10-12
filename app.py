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
