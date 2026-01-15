"""
main.py
End-to-end RAG pipeline using:
- DataLoader
- DataSplitter
- Embeddings
- Vectorstore (FAISS)
- Groq LLM RAG Chain
"""

# ============================================================
# IMPORTS
# ============================================================
from src.embedding import huggingface_embeddings
from src.vectorstore import load_vectorstore
from src.chain import create_rag_chain
from dotenv import load_dotenv
import os
load_dotenv()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("\n============================")
    print("🚀 Starting RAG Pipeline...")
    print("============================")

    embeddings = huggingface_embeddings(
        "sentence-transformers/all-MiniLM-L6-v2")

    # load existing vectorstore
    if os.path.exists("faiss_index"):
        vectorstore = load_vectorstore(
            embeddings, vectorstore_path="faiss_index")
        print("Loaded existing vectorstore from 'faiss_index'")
    else:
        print("❌ No existing vectorstore found. Please create one first.")

    # Build RAG chain
    rag_chain = create_rag_chain(vectorstore)

    while True:
        # Ask a question
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            print("👋 Exiting RAG Pipeline. Goodbye!")
            break

        # Display answer
        response = rag_chain.invoke({"input": query})
        print("\n🧠 AI Answer:")
        print(response["answer"])
        print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
