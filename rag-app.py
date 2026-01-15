"""
Streamlit RAG Chatbot Application
Simple and clean interface for querying your document collection
"""

import streamlit as st
import os
from dotenv import load_dotenv
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Title and description
st.title("🤖 AI RAG Chatbot")
st.markdown("Ask questions about your documents!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG chain


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


# Load the RAG chain
with st.spinner("🔄 Loading RAG system..."):
    rag_chain = initialize_rag_chain()

st.success("✅ RAG system ready!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]

                # Display answer
                st.markdown(answer)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ Error generating response: {e}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg})

# Sidebar with options
with st.sidebar:
    st.header("⚙️ Settings")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Info section
    st.subheader("ℹ️ About")
    st.info(
        """
        This chatbot uses RAG (Retrieval Augmented Generation) 
        to answer questions based on your document collection.
        
        **Features:**
        - Document retrieval with FAISS
        - Powered by Groq LLM
        - HuggingFace embeddings
        """
    )

    # Display stats
    if os.path.exists("faiss_index"):
        st.divider()
        st.subheader("📊 Stats")
        st.metric("Chat Messages", len(st.session_state.messages))
