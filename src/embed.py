from sentence_transformers import SentenceTransformer
import os
from typing import List
import faiss
import numpy as np
import pickle

def load_chunks(chunk_dir: str) -> List[str]:
    chunks = []
    files = sorted(os.listdir(chunk_dir))
    for fname in files:
        with open(os.path.join(chunk_dir, fname), "r", encoding="utf-8") as f:
            chunks.append(f.read())
    return chunks

def embed_chunks(model_name: str, chunk_dir: str, vectordb_path: str):
    model = SentenceTransformer(model_name)
    texts = load_chunks(chunk_dir)
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs(vectordb_path, exist_ok=True)
    faiss.write_index(index, f"{vectordb_path}/faiss.index")

    with open(f"{vectordb_path}/chunks.pkl", "wb") as f:
        pickle.dump(texts, f)

    print(f"✅ FAISS DB saved to '{vectordb_path}' with {len(texts)} chunks")
