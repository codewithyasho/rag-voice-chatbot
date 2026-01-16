"""
AI RAG Assistant - Text & Voice Chat (Improved)
Fixes applied:
1. Non-blocking asyncio handling
2. Unique audio file per assistant message
3. Correct audio playback in chat history
4. Conversation memory for RAG
5. Robust STT & TTS error handling
"""

import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from groq import Groq
import asyncio
import edge_tts
import base64
from datetime import datetime

# RAG imports
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain

# =========================
# ENV & CONFIG
# =========================
load_dotenv()

VOICE_INPUT_FILE = "voice_input.wav"

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"

# =========================
# RAG INITIALIZATION
# =========================


@st.cache_resource
def initialize_rag_chain():
    embeddings = huggingface_embeddings(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("faiss_index"):
        st.error("❌ FAISS index not found.")
        st.stop()

    vectorstore = load_vectorstore(
        embeddings, vectorstore_path="faiss_index"
    )

    return create_rag_chain(vectorstore)

# =========================
# VOICE UTILITIES
# =========================


def save_audio(audio_bytes, filename):
    with open(filename, "wb") as f:
        f.write(audio_bytes)


def speech_to_text(audio_file):
    try:
        client = Groq()
        with open(audio_file, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(audio_file, f.read()),
                model="whisper-large-v3-turbo",
                temperature=0,
            )
        return transcription.text
    except Exception as e:
        return None


async def _tts_async(text, output_file):
    voice = "hi-IN-MadhurNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def text_to_speech(text, output_file):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_async(text, output_file))
        loop.close()
        return True
    except Exception:
        return False


def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

# =========================
# RAG WITH MEMORY
# =========================


def process_rag_query(query):
    try:
        history = []
        for msg in st.session_state.messages[-6:]:
            history.append(f"{msg['role']}: {msg['content']}")

        full_query = "\n".join(history) + f"\nuser: {query}"
        response = rag_chain.invoke({"input": full_query})
        return response["answer"]
    except Exception as e:
        return "❌ RAG processing failed."


# =========================
# UI
# =========================
st.title("🤖 AI RAG Assistant")
st.markdown("*Text & Voice enabled RAG chatbot*")

with st.spinner("Loading RAG system..."):
    rag_chain = initialize_rag_chain()

st.success("RAG Ready")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Input Mode",
        ["💬 Text Chat", "🎤 Voice Chat"]
    )

    st.session_state.input_mode = "voice" if "Voice" in mode else "text"

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.metric("Messages", len(st.session_state.messages))
    st.metric("Mode", st.session_state.input_mode.upper())

# =========================
# VOICE MODE
# =========================
if st.session_state.input_mode == "voice":

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        key=f"rec_{len(st.session_state.messages)}"
    )

    if audio:
        with st.spinner("Processing voice..."):
            save_audio(audio["bytes"], VOICE_INPUT_FILE)

            text = speech_to_text(VOICE_INPUT_FILE)
            if not text:
                st.warning("Speech-to-text failed.")
                st.stop()

            st.session_state.messages.append(
                {"role": "user", "content": text}
            )

            answer = process_rag_query(text)

            audio_file = f"audio_{datetime.now().timestamp()}.mp3"
            tts_ok = text_to_speech(answer, audio_file)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "audio_path": audio_file if tts_ok else None
                }
            )

            if tts_ok:
                st.markdown(autoplay_audio(audio_file),
                            unsafe_allow_html=True)

# =========================
# TEXT MODE
# =========================
else:
    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        answer = process_rag_query(prompt)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"], format="audio/mp3")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center><small>Powered by LangChain, FAISS, Groq & Edge TTS</small></center>",
    unsafe_allow_html=True
)
