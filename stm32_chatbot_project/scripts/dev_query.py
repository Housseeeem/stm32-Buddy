# scripts/rag_query.py

import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# Load FAISS + metadata
index = faiss.read_index("faiss_index/stm32.index")
with open("faiss_index/meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load StarCoder2-3B model (trained for multiple languages including C)
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
model = AutoModelForCausalLM.from_pretrained(
    "bigcode/starcoder2-3b",
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Function to search relevant context using FAISS
def search_context(query, k=5):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [docs[i] for i in indices[0]]

# RAG-style question answering
def ask_rag_question(query):
    top_contexts = search_context(query)
    context = "\n".join(top_contexts)

    prompt = (
        f"Tu es un assistant expert STM32 et programmation embarquée en langage C.\n"
        f"Voici quelques extraits de documentation technique :\n\n"
        f"{context}\n\n"
        f"Question : {query}\n"
        f"Réponse (en C embarqué si possible) :"
    )

    result = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    # return result[0]["generated_text"]
    full_output = result[0]["generated_text"]

    # ✅ Extraire seulement la première réponse
    if "Réponse" in full_output:
        # Prendre ce qui vient après "Réponse" jusqu'à la prochaine "Question" (s'il y en a)
        response_part = full_output.split("Réponse")[1]
        clean_response = response_part.split("Question")[0].strip(" :\n")
    else:
        clean_response = full_output.strip()

    return f"Réponse : {clean_response}"

# Console mode
if __name__ == "__main__":
    question = input("Ask your STM32 question: ")
    answer = ask_rag_question(question)
    print("\n🤖", answer)
