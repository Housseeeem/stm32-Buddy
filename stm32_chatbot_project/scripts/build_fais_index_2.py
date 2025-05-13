# scripts/build_faiss_index.py
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the data
try:
    with open("../data/stm32_clean_data.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
except FileNotFoundError:
    print("❌ Data file not found. Ensure stm32_clean_data.json exists in the correct path.")
    exit(1)

# Choose a small embedding model
print("🔎 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")

# Embed the chunks and include metadata
texts = [{"content": chunk["content"], "source": chunk["source"], "page": chunk.get("page", "N/A")} for chunk in chunks]
contents_only = [chunk["content"] for chunk in chunks]  # Separate content for embedding
embeddings = model.encode(contents_only, show_progress_bar=True)

# FAISS index creation
print("⚙️ Creating FAISS index...")
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Create output folder if it doesn't exist
os.makedirs("faiss_index", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "faiss_index/stm32_faiss.index")
print(f"✅ FAISS index created with {index.ntotal} vectors.")

# Save metadata
try:
    with open("faiss_index/stm32_meta.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print("✅ Metadata saved successfully.")
except Exception as e:
    print(f"❌ Error saving metadata: {e}")

# Save embeddings (optional, for debugging)
np.save("faiss_index/stm32_embeddings.npy", np.array(embeddings))
print("✅ Embeddings saved.")

print("🎉 FAISS index and metadata preparation complete!")