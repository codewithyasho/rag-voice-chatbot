
"""
AI Voice Assistant Chatbot - Streamlit Interface
Flow: Audio Recording -> Speech-to-Text -> LLM Processing -> Text-to-Speech -> Play Audio
"""

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq
from dotenv import load_dotenv
import asyncio
import edge_tts
import os
import base64
from datetime import datetime

# Load environment variables
load_dotenv()

# File paths
VOICE_OUTPUT_FILE = "voice-output.wav"
HUMAN_TEXT_FILE = "human-text.txt"
LLM_TEXT_FILE = "llm-text.txt"
AUDIO_OUTPUT_FILE = "audio_output.mp3"

# Page configuration
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎤",
    layout="centered"
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


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

    # Save transcription
    with open(HUMAN_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(transcription.text)

    return transcription.text


def process_with_llm(user_query):
    """Process text with LLM"""
    try:
        client = Groq()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": f"You are a helpful voice assistant. Answer the following query in short and in a concise manner: {user_query}"
                }
            ],
            temperature=0.2,
        )

        result = completion.choices[0].message.content
        # print(result)

        # Save LLM response
        with open(LLM_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(result)

        return result
    except Exception as e:
        return f"Error: {str(e)}. Make sure Ollama is running and the model is installed."


async def text_to_speech_async(text):
    """Convert text to speech"""
    VOICE = "hi-IN-MadhurNeural"
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


# App UI
st.title("🎤 AI Voice Assistant Chatbot")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("**Pipeline:**\n1. 🎙️ Record Audio\n2. 📝 Speech-to-Text\n3. 🤖 LLM Processing\n4. 🔊 Text-to-Speech\n5. ▶️ Play Response")

    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.rerun()

# Main content
st.subheader("🎙️ Record Your Voice")
st.markdown("Click the button below to start recording your question:")

# Audio recorder
audio_data = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=False,
    key="voice_recorder"
)

# Process audio when recorded
if audio_data:
    with st.spinner("🎧 Processing your voice..."):
        try:
            # Step 1: Save audio from bytes
            save_audio_from_bytes(audio_data["bytes"], VOICE_OUTPUT_FILE)
            st.success("✅ Audio recorded successfully!")

            # Step 2: Speech to text
            with st.spinner("📝 Transcribing..."):
                transcription = speech_to_text(VOICE_OUTPUT_FILE)

            st.info(f"**You said:** {transcription}")

            # Step 3: Process with LLM
            with st.spinner("🤖 Thinking..."):
                llm_response = process_with_llm(transcription)

            st.success(f"**Assistant:** {llm_response}")

            # Step 4: Text to speech
            with st.spinner("🔊 Generating voice response..."):
                text_to_speech(llm_response)

            # Step 5: Play audio
            st.success("✅ Playing audio response...")
            st.audio(AUDIO_OUTPUT_FILE, format="audio/mp3")

            # Auto-play audio (optional)
            st.markdown(autoplay_audio(AUDIO_OUTPUT_FILE),
                        unsafe_allow_html=True)

            # Add to conversation history
            st.session_state.conversation_history.append({
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "user": transcription,
                "assistant": llm_response
            })

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display conversation history
if st.session_state.conversation_history:
    st.markdown("---")
    st.subheader("💬 Conversation History")

    for idx, conv in enumerate(reversed(st.session_state.conversation_history)):
        with st.expander(f"🕒 {conv['timestamp']} - Conversation {len(st.session_state.conversation_history) - idx}"):
            st.markdown(f"**👤 You:** {conv['user']}")
            st.markdown(f"**🤖 Assistant:** {conv['assistant']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Powered by Groq Whisper, Ollama, and Edge TTS</small>
    </div>
    """,
    unsafe_allow_html=True
)
