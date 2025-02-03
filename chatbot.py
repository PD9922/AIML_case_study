import pickle
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "/Users/excalibur/Documents/ANACONDA/AIML/faiss_index.pkl"
TOP_K = 5
OPENAI_API_KEY = ""

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
openai.api_key = OPENAI_API_KEY

def load_vector_store():
    with open(FAISS_INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["metadata"], data["corpus"]

def get_relevant_chunks(query, index, corpus, top_k=TOP_K):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [corpus[idx] for idx in indices[0]]
    return retrieved_chunks

def generate_answer(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"Using the following context from Ubuntu documentation:\n\n{context_text}\n\nAnswer the following question:\n\n{query}\n"
    
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stop=None
    )
    answer = response.choices[0].message.content
    return answer

def answer_query(query):
    index, metadata, corpus = load_vector_store()
    context_chunks = get_relevant_chunks(query, index, corpus)
    answer = generate_answer(query, context_chunks)
    return answer

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == "quit":
            break
        try:
            answer = answer_query(user_query)
            print("\nAnswer:", answer, "\n")
        except Exception as e:
            print("Error during processing:", str(e))