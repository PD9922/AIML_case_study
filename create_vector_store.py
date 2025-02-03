import os
import re
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = "/Users/excalibur/Documents/ANACONDA/AIML/demo_bot_data/demo_bot_data/ubuntu-docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600  
CHUNK_OVERLAP = 50

def load_markdown_files(directory):
    docs = {}
    for fname in os.listdir(directory):
        if fname.endswith(".md"):
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_vector_store():
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    docs = load_markdown_files(DOCS_DIR)
    
    corpus = []
    metadata = []
    for fname, text in docs.items():
        text = clean_text(text)
        chunks = chunk_text(text)
        for chunk in chunks:
            corpus.append(chunk)
            metadata.append({"source": fname})
    
    print(f"Total chunks created: {len(corpus)}")
    
    embeddings = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump({
            "index": index,
            "metadata": metadata,
            "corpus": corpus,
            "embeddings": embeddings
        }, f)
    print("Vector store created and saved as faiss_index.pkl")

if __name__ == "__main__":
    build_vector_store()