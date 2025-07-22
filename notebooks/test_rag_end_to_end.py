# notebooks/test_rag_end_to_end.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import DocumentRetriever
from src.generator import stream_response

retriever = DocumentRetriever("vectordb")

query = "Does the document allow message scanning for privacy violations?"
results = retriever.retrieve(query, top_k=3)

prompt = retriever.format_prompt(query, results)

stream_response(prompt)
