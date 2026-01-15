"""
AI RAG Assistant - Text & Voice Chat
Combines RAG-powered document Q&A with text and voice interaction capabilities
Flow: User Input (Text/Voice) -> Speech-to-Text (if voice) -> RAG Processing -> Response -> Text-to-Speech (if voice mode)
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

# Import RAG components
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain

# Load environment variables
load_dotenv()

# File paths for voice processing
VOICE_OUTPUT_FILE = "voice-output.wav"
AUDIO_OUTPUT_FILE = "audio_output.mp3"

# Page configuration
st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "text"  # Default to text mode


# ============================================================
# RAG INITIALIZATION
# ============================================================
@st.cache_resource
def initialize_rag_chain():
    """Initialize and cache the RAG chain"""
    try:
        # Load embeddings
        embeddings = huggingface_embeddings(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Check if vectorstore exists
        if not os.path.exists("faiss_index"):
            st.error(
                "⚠️ Vector store not found! Please create the FAISS index first.")
            st.stop()

        # Load vectorstore
        vectorstore = load_vectorstore(
            embeddings, vectorstore_path="faiss_index")

        # Create RAG chain
        rag_chain = create_rag_chain(vectorstore)

        return rag_chain

    except Exception as e:
        st.error(f"❌ Error initializing RAG chain: {e}")
        st.stop()


# ============================================================
# VOICE PROCESSING FUNCTIONS
# ============================================================
def save_audio_from_bytes(audio_bytes, filename):
    """Save audio bytes to a WAV file"""
    with open(filename, "wb") as f:
        f.write(audio_bytes)


def speech_to_text(audio_file):
    """Convert speech to text using Whisper"""
    client = Groq()

    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_file, file.read()),
            model="whisper-large-v3-turbo",
            temperature=0,
            response_format="verbose_json",
        )

    return transcription.text


async def text_to_speech_async(text):
    """Convert text to speech"""
    VOICE = "hi-IN-MadhurNeural"  # Change voice as needed
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(AUDIO_OUTPUT_FILE)


def text_to_speech(text):
    """Wrapper for async text-to-speech function"""
    asyncio.run(text_to_speech_async(text))


def autoplay_audio(file_path):
    """Generate HTML to autoplay audio"""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    return audio_html


# ============================================================
# RAG QUERY FUNCTION
# ============================================================
def process_rag_query(query):
    """Process query through RAG chain"""
    try:
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================
# UI LAYOUT
# ============================================================

# Title
st.title("🤖 AI RAG Assistant")
st.markdown("*Chat with your documents using Text or Voice*")

# Load RAG chain
with st.spinner("🔄 Loading RAG system..."):
    rag_chain = initialize_rag_chain()

st.success("✅ RAG system ready!")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    # Input mode selector
    st.subheader("🎛️ Input Mode")
    input_mode = st.radio(
        "Choose your input method:",
        ["💬 Text Chat", "🎤 Voice Chat"],
        key="mode_selector"
    )

    # Update input mode in session state
    if "Text" in input_mode:
        st.session_state.input_mode = "text"
    else:
        st.session_state.input_mode = "voice"

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Pipeline info
    if st.session_state.input_mode == "voice":
        st.subheader("🔄 Voice Pipeline")
        st.info(
            """
            1. 🎙️ Record Audio
            2. 📝 Speech-to-Text
            3. 🔍 RAG Retrieval
            4. 🤖 LLM Response
            5. 🔊 Text-to-Speech
            6. ▶️ Play Audio
            """
        )
    else:
        st.subheader("🔄 Text Pipeline")
        st.info(
            """
            1. 💬 Type Query
            2. 🔍 RAG Retrieval
            3. 🤖 LLM Response
            4. 📄 Display Answer
            """
        )

    st.divider()

    # Stats
    st.subheader("📊 Stats")
    st.metric("Chat Messages", len(st.session_state.messages))
    st.metric("Current Mode", st.session_state.input_mode.upper())


# ============================================================
# MAIN CHAT INTERFACE
# ============================================================

# ============================================================
# VOICE MODE - PLACE RECORDER AT TOP
# ============================================================
if st.session_state.input_mode == "voice":
    st.markdown("---")
    
    # Create a fixed container for voice input at the top
    voice_container = st.container()
    
    with voice_container:
        st.subheader("🎙️ Voice Input")
        st.markdown("Click the button below to record your question:")
        
        # Audio recorder with dynamic key
        audio_data = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,  # Changed to True to prevent conflicts
            use_container_width=False,
            key=f"voice_recorder_{len(st.session_state.messages)}"  # Dynamic key
        )
    
    st.markdown("---")
    
    # Process audio when recorded
    if audio_data:
        with st.spinner("🎧 Processing your voice..."):
            try:
                # Step 1: Save audio from bytes
                save_audio_from_bytes(audio_data["bytes"], VOICE_OUTPUT_FILE)

                # Step 2: Speech to text
                with st.spinner("📝 Transcribing..."):
                    transcription = speech_to_text(VOICE_OUTPUT_FILE)

                # Display user message
                with st.chat_message("user"):
                    st.markdown(f"🎤 **Voice:** {transcription}")

                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"🎤 {transcription}"
                })

                # Step 3: Process with RAG
                with st.spinner("🔍 Searching documents..."):
                    rag_response = process_rag_query(transcription)

                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(rag_response)

                    # Step 4: Text to speech
                    with st.spinner("🔊 Generating voice response..."):
                        text_to_speech(rag_response)

                    # Step 5: Play audio
                    st.success("✅ Playing audio response...")
                    st.audio(AUDIO_OUTPUT_FILE, format="audio/mp3")

                    # Auto-play audio
                    st.markdown(autoplay_audio(AUDIO_OUTPUT_FILE),
                                unsafe_allow_html=True)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": rag_response,
                    "audio": True
                })

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
    # Display chat history for voice mode
    if st.session_state.messages:
        st.subheader("💬 Conversation History")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # If there's audio output, display audio player
                if "audio" in message and message["audio"]:
                    if os.path.exists(AUDIO_OUTPUT_FILE):
                        st.audio(AUDIO_OUTPUT_FILE, format="audio/mp3")


# ============================================================
# TEXT MODE
# ============================================================
else:
    # Display chat history for text mode
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Text input at bottom
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = process_rag_query(prompt)
                
                # Display answer
                st.markdown(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>🚀 Powered by LangChain, FAISS, Groq, HuggingFace & Edge TTS</small>
    </div>
    """,
    unsafe_allow_html=True
)
