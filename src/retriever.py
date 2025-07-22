# src/retriever.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, vectordb_path: str, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(f"{vectordb_path}/faiss.index")
        with open(f"{vectordb_path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), top_k)
        results = [(self.chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

    def format_prompt(self, query: str, retrieved_chunks: list) -> str:
        context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, (chunk, _) in enumerate(retrieved_chunks)])
        prompt = f"""You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question: {query}
Answer:"""
        return prompt
