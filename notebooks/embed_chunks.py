import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.embed import embed_chunks

embed_chunks(
    model_name="all-MiniLM-L6-v2",   # fast and solid
    chunk_dir="chunks",
    vectordb_path="vectordb"
)
