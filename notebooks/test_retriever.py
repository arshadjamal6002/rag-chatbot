import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.retriever import DocumentRetriever

retriever = DocumentRetriever(vectordb_path="vectordb")

query = "What does the document say about data privacy?"
results = retriever.retrieve(query)

print("Top Retrieved Chunks:\n")
for i, (chunk, score) in enumerate(results):
    print(f"[{i+1}] Score: {score:.4f}")
    print(chunk[:300], "...\n")

prompt = retriever.format_prompt(query, results)
print("-----\nFinal Prompt Sent to LLM:\n")
print(prompt)
