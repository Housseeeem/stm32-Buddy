import json
import faiss
import numpy as np
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer

# Load FAISS index + metadata
index = faiss.read_index("faiss_index/stm32.index")
with open("faiss_index/meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# SentenceTransformer for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Search top K documents using FAISS
def search_context(query, k=5):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [docs[i] for i in indices[0]]

# RAG-style QA using Ollama
def ask_rag_question(query):
    top_contexts = search_context(query)
    context = "\n".join(top_contexts)

    prompt = (
        f"<s>[INST] Tu es un assistant expert en STM32 et programmation embarquée en langage C.\n"
        f"Voici des extraits de documentation technique :\n\n"
        f"{context}\n\n"
        f"Question : {query}\n"
        f"Réponds avec précision, en C embarqué si possible. [/INST]"

    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:latest",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3,
            "num_predict": 100
        }
    )

    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return f"❌ Erreur Ollama : {response.text}"

# Interface Gradio
iface = gr.Interface(
    fn=ask_rag_question,
    inputs=gr.Textbox(lines=3, placeholder="Pose ta question sur STM32..."),
    outputs="text",
    title="🔧🤖 Assistant STM32 (IA)",
    description="Un assistant intelligent basé sur l'IA pour répondre à tes questions sur la programmation embarquée en STM32 en langage C."
)

iface.launch(share=True)

if __name__ == "__main__":
    iface.launch()
