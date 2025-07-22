# notebooks/run_preprocess.py or inside a Jupyter notebook
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.preprocess import preprocess_document

doc_path = "data/document.pdf"   # Update to your actual file
chunk_path = "chunks"

preprocess_document(doc_path, chunk_path)
