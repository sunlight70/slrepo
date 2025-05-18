import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader

import torch
torch.classes.__path__ = []
import streamlit as st

# Change this to your actual Ollama install path
OLLAMA_PATH = r"C:\Users\Shawn\AppData\Local\Programs\Ollama\ollama.exe"

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
    return text

# Split text into chunks
def split_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Embed chunks and create FAISS index
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Search FAISS index
def search_index(query, index, chunks, top_k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in I[0]]

# Run local model via Ollama with full path
##def generate_answer_ollama(context, question, model="llama2:7b"):
def generate_answer_ollama(context, question, model="llama3"):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately.

Context:
{context}

Question: {question}
Answer:"""
    try:
        result = subprocess.run([OLLAMA_PATH, "run", model, prompt], capture_output=True, text=True, encoding='utf-8', timeout=300)
        print("Ollama stdout:", result.stdout)
        print("Ollama stderr:", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Ollama returned error code {result.returncode}: {result.stderr}")
        return result.stdout.strip()
    except Exception as e:
        st.error(f"Error running Ollama: {e}")
        return "Failed to generate answer."


# Streamlit UI
st.set_page_config(page_title="KB Chatbot (Ollama Windows)", layout="wide")
st.title("📚KB Chatbot")
#upload any file of your choice
#uploaded_file = st.file_uploader("Upload a PDF Guide", type="pdf")

#from your local directory
file_path = "./Guide.pdf"
with open(file_path, "rb") as f:
        uploaded_file = f  # Mimic file_uploader's output (a file-like object)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        index, embeddings = create_faiss_index(chunks)
        st.success(f"Indexed {len(chunks)} chunks from PDF.")

    query = st.text_input("Ask your question:")

    if query:
        with st.spinner("Generating answer..."):
            relevant_chunks = search_index(query, index, chunks)
            context = "\n\n".join(relevant_chunks)
            answer = generate_answer_ollama(context, query)
            st.markdown(f"### ✅ Answer:\n{answer}")

def generate_answer_ollama(context, question, model="llama3"):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately.

Context:
{context}

Question: {question}
Answer:"""
    result = subprocess.run([OLLAMA_PATH, "run", model, prompt], capture_output=True, text=True)
    print("Ollama stdout:", result.stdout)
    print("Ollama stderr:", result.stderr)
    return result.stdout.strip()
