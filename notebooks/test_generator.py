import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.retriever import DocumentRetriever
from src.generator import generate_answer_hf

retriever = DocumentRetriever(vectordb_path="vectordb")
query = "What does this document say about user privacy?"

top_chunks = retriever.retrieve(query)
prompt = retriever.format_prompt(query, top_chunks)

generate_answer_hf(prompt, stream=True)
