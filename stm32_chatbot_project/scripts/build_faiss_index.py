# scripts/build_faiss_index.py
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the data
with open("../data/stm32_clean_data.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Choose a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed the chunks
texts = [chunk['content'] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Create output folder if it doesn't exist
os.makedirs("faiss_index", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "faiss_index/stm32.index")

# Save metadata
with open("faiss_index/meta.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("✅ FAISS index created.")
