import streamlit as st
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----- Helper Functions -----
@st.cache_data
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@st.cache_data
def split_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

@st.cache_data
def get_embedding(text, model="text-embedding-ada-002"):
    result = openai.Embedding.create(input=[text], model=model)
    return result["data"][0]["embedding"]

@st.cache_data
def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]

@st.cache_resource
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def search_index(query, index, chunks, top_k=3):
    query_embedding = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following guide information to answer the user's question accurately.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# ----- Streamlit App -----
st.set_page_config(page_title="Knowledge Guide Chatbot", layout="wide")

st.title("📚 Knowledge-Based Chatbot (PDF Q&A)")
st.markdown("Ask questions based on the uploaded PDF guide.")

uploaded_file = st.file_uploader("Upload your PDF guide", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF... This may take a minute."):
        guide_text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(guide_text)
        embeddings = embed_chunks(chunks)
        index = create_faiss_index(embeddings)
        st.success(f"PDF processed! {len(chunks)} chunks indexed.")

    user_question = st.text_input("Ask your question here:")

    if user_question:
        with st.spinner("Searching for answer..."):
            relevant_chunks = search_index(user_question, index, chunks)
            answer = generate_answer(user_question, relevant_chunks)
            st.markdown(f"### ✅ Answer:\n{answer}")
