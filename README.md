# 🤖 AI RAG Assistant - Text & Voice Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A multimodal RAG (Retrieval Augmented Generation) chatbot designed to answer questions from your documents using both text and voice interactions.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Configuration](#-configuration)

</div>

---

## Overview

This project implements a sophisticated **RAG (Retrieval Augmented Generation)** system that allows users to interact with their documents through both **text and voice**. Built with LangChain, FAISS, Groq LLM, and Streamlit, it provides an intuitive interface for document-based question answering with multimodal input/output capabilities.

### Why RAG?

RAG combines the power of:

- **Information Retrieval**: Find relevant context from your documents
- **Language Models**: Generate accurate, contextual responses
- **Real-time Processing**: Get instant answers without retraining models

---

## ✨ Features

### Core Features

- 🔍 **Semantic Document Search** - FAISS vector store with cosine similarity
- 🤖 **Groq LLM Integration** - Fast and accurate response generation
- 💬 **Text Chat Interface** - Traditional text-based Q&A
- 🎤 **Voice Input** - Record questions using your microphone
- 🔊 **Voice Output** - Text-to-speech responses (toggle on/off)
- 📄 **Multi-format Support** - PDF, DOCX, TXT, CSV, Excel, PPT, JSON, and web pages
- 🧠 **Context-Aware Responses** - Retrieves relevant document chunks
- 💾 **Persistent Vector Store** - Load existing indices or create new ones
- 🎨 **Modern UI** - Clean, intuitive Streamlit interface

### Advanced Features

- **MMR Retrieval** - Maximum Marginal Relevance for diverse results
- **Dynamic Chat History** - Conversation tracking
- **Normalized Embeddings** - Optimized for cosine similarity
- **Batch Processing** - Efficient document embedding
- **Error Handling** - Comprehensive error management
- **Device Detection** - Auto-detect CUDA for GPU acceleration

---

## 🎬 Demo

### Text Mode

```
User: "What is machine learning?"
Assistant: "Based on the provided information, machine learning is a subset
of artificial intelligence (AI) that enables systems to learn from data..."
```

### Voice Mode

1. Click "🎙️ Start Recording"
2. Speak your question
3. Receive transcribed text + AI response
4. Listen to voice output (if enabled)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Streamlit Web App)                       │
└────────────┬──────────────────────────────┬─────────────────┘
             │                              │
      ┌──────▼──────┐              ┌────────▼────────┐
      │ Text Input  │              │  Voice Input    │
      └──────┬──────┘              └────────┬────────┘
             │                              │
             │                         ┌────▼─────┐
             │                         │ Whisper  │
             │                         │   STT    │
             │                         └────┬─────┘
             │                              │
        ┌────▼──────────────────────────────▼────┐
        │         Query Processing                │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │         Document Retrieval              │
        │    (FAISS Vector Store Search)          │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │      Context + Query → Groq LLM         │
        │         (Response Generation)           │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────▼────────────────────────┐
        │           Response Display              │
        │        (Text + Optional Audio)          │
        └─────────────────────────────────────────┘
                         │
                    ┌────▼─────┐
                    │ Edge TTS │
                    │   (TTS)  │
                    └──────────┘
```

---

## 📁 Project Structure

```
RAHUL_RAG_voice/
│
├── app.py                      # Main Streamlit application (Production)
├── main.py                     # Terminal-based RAG pipeline
├── rag-app.py                  # Text-only chatbot
├── voice-chatbot-app.py        # Voice-only chatbot
│
├── src/                        # Core RAG components
│   ├── __init__.py
│   ├── chain.py                # RAG chain creation (LLM + Retriever)
│   ├── dataloader.py           # Document loaders (PDF, DOCX, etc.)
│   ├── datasplitter.py         # Text chunking
│   ├── embedding.py            # HuggingFace/Ollama embeddings
│   ├── prompt.py               # LLM prompt templates
│   ├── utils.py                # Utility functions
│   └── vectorstore.py          # FAISS operations (create/load/add)
│
├── data/                       # Your documents directory
│   └── documents.json          # Sample JSON data
│
├── faiss_index/                # FAISS vector store (auto-generated)
│   └── index.faiss
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- Groq API Key ([Get one here](https://console.groq.com/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-rag-assistant.git
cd ai-rag-assistant
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Prepare Your Documents

Place your documents in the `data/` folder:

```bash
mkdir -p data
# Add your PDF, DOCX, TXT, CSV, Excel files here
```

### Step 6: Create Vector Store (First Time Only)

Run this to index your documents:

```python
# Create a script: create_index.py
from src.dataloader import process_all_pdfs, process_all_texts
from src.datasplitter import split_docs
from src.embedding import huggingface_embeddings
from src.vectorstore import create_vectorstore

# Load documents
documents = []
documents.extend(process_all_pdfs("data"))
documents.extend(process_all_texts("data"))

# Split into chunks
chunks = split_docs(documents)

# Create embeddings
embeddings = huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# Create and save vector store
vectorstore = create_vectorstore(chunks, embeddings)
print("Vector store created successfully!")
```

Run it:

```bash
python create_index.py
```

---

## ⚙️ Configuration

### LLM Configuration

Edit `src/chain.py` to change the LLM model:

```python
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # Change model here
    temperature=0.3,              # Adjust temperature
)
```

### Embedding Model

Edit `app.py` to change embedding model:

```python
embeddings = huggingface_embeddings(
    "sentence-transformers/all-MiniLM-L6-v2"  # Change model here
)
```

**Popular Embedding Models:**

- `sentence-transformers/all-MiniLM-L6-v2` (Fast, 384 dims)
- `sentence-transformers/all-mpnet-base-v2` (Better quality, 768 dims)
- `BAAI/bge-small-en-v1.5` (Optimized for retrieval)

### Voice Configuration

Edit voice settings in `app.py`:

```python
# Text-to-Speech Voice
VOICE = "hi-IN-MadhurNeural"  # Change to your preferred voice
```

### Retrieval Configuration

Edit `src/chain.py`:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",        # Options: "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 7}    # Number of documents to retrieve
)
```

---

## 💻 Usage

### Run the Main Application

```bash
streamlit run app.py
```

Open browser at: `http://localhost:8501`

### Alternative Interfaces

#### 1. Terminal-based RAG (No GUI)

```bash
python main.py
```

#### 2. Text-only Chat

```bash
streamlit run rag-app.py
```

#### 3. Voice-only Chat

```bash
streamlit run voice-chatbot-app.py
```

---

## 🔧 How It Works

### 1. Document Processing

```python
# Load → Split → Embed → Store
Documents → Chunks → Vectors → FAISS Index
```

### 2. Query Processing

```python
# Input → Retrieve → Generate → Output
User Query → Relevant Docs → LLM Response → Answer
```

### 3. Voice Pipeline

```python
# Record → Transcribe → Process → Synthesize → Play
Voice → Text → RAG → Response Text → Audio
```

---

## 🧩 Components

### Core Modules

#### `dataloader.py`

Loads documents from various formats:

- `process_all_pdfs()` - PDF files
- `process_all_texts()` - Text files
- `process_all_excels()` - Excel files
- `process_all_word_docs()` - Word documents
- `process_all_csvs()` - CSV files
- `process_all_webpages()` - Web URLs

#### `datasplitter.py`

Chunks documents for optimal retrieval:

- Chunk size: 1200 characters
- Overlap: 200 characters
- Recursive splitting strategy

#### `embedding.py`

Generates vector embeddings:

- HuggingFace Embeddings (default)
- Ollama Embeddings (alternative)
- GPU acceleration support

#### `vectorstore.py`

Manages FAISS vector store:

- `create_vectorstore()` - Create new index
- `load_vectorstore()` - Load existing index
- `load_and_add_new_docs()` - Update index

#### `chain.py`

Creates RAG chain:

- Retriever setup
- LLM initialization
- Prompt template
- Chain construction

#### `prompt.py`

LLM prompt templates:

- General purpose prompt
- Custom system instructions

---

## 📄 Supported Document Types

| Format     | Extension | Loader                         |
| ---------- | --------- | ------------------------------ |
| PDF        | `.pdf`    | PyMuPDFLoader                  |
| Text       | `.txt`    | TextLoader                     |
| Word       | `.docx`   | UnstructuredWordDocumentLoader |
| Excel      | `.xlsx`   | StructuredExcelLoader          |
| CSV        | `.csv`    | CSVLoader                      |
| PowerPoint | `.pptx`   | UnstructuredPowerPointLoader   |
| JSON       | `.json`   | JSONLoader                     |
| Web Pages  | URL       | WebBaseLoader                  |

---

## 🔑 API Keys

### Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up / Log in
3. Navigate to API Keys
4. Create a new key
5. Add to `.env` file

### Optional: Hugging Face Token

For private models:

```env
HUGGINGFACE_TOKEN=your_token_here
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. FAISS Index Not Found

```bash
Error: Vector store not found!
```

**Solution:** Create the vector store first using `create_index.py`

#### 2. GROQ_API_KEY Not Set

```bash
Error: API key not found
```

**Solution:** Add `GROQ_API_KEY` to `.env` file

#### 3. Audio Device Issues

```bash
Error: No audio device found
```

**Solution:** Check microphone permissions and device settings

#### 4. Out of Memory

```bash
Error: CUDA out of memory
```

**Solution:**

- Use CPU instead: Set `device='cpu'` in `embedding.py`
- Reduce batch size: `batch_size=16`
- Use smaller embedding model

#### 5. Slow Performance

**Solutions:**

- Use GPU for embeddings (if available)
- Reduce `chunk_size` in `datasplitter.py`
- Lower `k` value in retriever
- Use `faiss-gpu` instead of `faiss-cpu`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 TODO / Roadmap

- [ ] Add support for more LLM providers (OpenAI, Anthropic)
- [ ] Implement chat history persistence (SQLite/JSON)
- [ ] Add document upload via UI
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] API endpoint (FastAPI)
- [ ] Authentication system
- [ ] Advanced filters (date, author, document type)
- [ ] Export chat history
- [ ] Citation/source tracking
- [ ] Streaming responses
- [ ] Mobile-responsive UI

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Groq Documentation](https://console.groq.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace Embeddings](https://huggingface.co/models?pipeline_tag=sentence-similarity)

---

## 🙏 Acknowledgments

- **LangChain** - For the RAG framework
- **Groq** - For fast LLM inference
- **HuggingFace** - For embedding models
- **Streamlit** - For the web interface
- **FAISS** - For efficient vector search
- **Edge TTS** - For text-to-speech

---

<div align="center">

**Made with ❤️ using LangChain, FAISS, Groq, and Streamlit**

[⬆ Back to Top](#-ai-rag-assistant---text--voice-chatbot)

</div>
