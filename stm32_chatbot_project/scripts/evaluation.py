import json
import faiss
import time
import numpy as np
import requests
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
index = faiss.read_index("faiss_index/stm32.index")
with open("faiss_index/meta.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Search top-K documents using FAISS
def search_context(query, k=5):
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [docs[i] for i in indices[0]]

# Ask question using RAG prompt and Ollama
def ask_rag_question(query):
    top_contexts = search_context(query)
    context = "\n---\n".join(top_contexts)

    prompt = (
        f"[CONTEXTES EXTRAITS DE DOCUMENTATION]\n"
        f"{context}\n\n"
        f"[INSTRUCTION]\n"
        f"En te basant uniquement sur les extraits ci-dessus, réponds à la question suivante :\n"
        f"{query}\n\n"
        f"Réponse en langage C embarqué si possible. N'invente rien si l'information n'est pas présente dans le contexte."
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.2,
            "num_predict": 300
        }
    )

    if response.status_code == 200:
        return response.json()["response"].strip(), top_contexts  # <-- maintenant deux valeurs
    else:
        return f"❌ Erreur Ollama : {response.text}", top_contexts


# def ask_rag_question(query):
#     top_contexts = search_context(query)
#     context = "\n".join(top_contexts)
#
#     prompt = (
#         f"Tu es un assistant STM32 expert en C embarqué.\n"
#         f"Voici des extraits de documentation technique :\n\n"
#         f"{context}\n\n"
#         f"Réponds précisément à la question suivante en utilisant exclusivement les extraits ci-dessus.\n"
#         f"Question : {query}\n"
#         f"Réponse :"
#     )
#
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "mistral:latest",
#             "prompt": prompt,
#             "stream": False,
#             "temperature": 0.3,
#             "num_predict": 100
#         }
#     )
#
#     if response.status_code == 200:
#         return response.json()["response"].strip(), top_contexts
#     else:
#         return f"❌ Erreur Ollama : {response.text}", top_contexts

# Evaluate similarity (for relevance and context match)
def is_relevant(response, reference, threshold=0.7):
    ratio = SequenceMatcher(None, response.lower(), reference.lower()).ratio()
    return ratio >= threshold

def context_match(response, snippets, threshold=0.6):
    for snippet in snippets:
        ratio = SequenceMatcher(None, response.lower(), snippet.lower()).ratio()
        if ratio >= threshold:
            return True
    return False

# Time + evaluate a single question
def timed_ask(query):
    start = time.time()
    response, top_k = ask_rag_question(query)
    latency = time.time() - start
    return response, top_k, latency

# Main batch evaluation
def evaluate_all(queries, references):
    latencies = []
    relevance_hits = 0
    context_hits = 0
    responses = []

    for i, (q, ref) in enumerate(zip(queries, references)):
        print(f"\n🟡 Question {i+1}: {q}")
        response, top_k, latency = timed_ask(q)
        print(f"🟢 Réponse: {response}")
        print(f"⏱️ Latence: {round(latency, 2)} sec")

        latencies.append(latency)
        responses.append(response)

        if is_relevant(response, ref):
            relevance_hits += 1
            print("✅ Réponse jugée pertinente.")
        else:
            print("❌ Réponse non pertinente.")

        if context_match(response, top_k):
            context_hits += 1
            print("🔎 Réponse correspond au contexte.")
        else:
            print("⚠️ Réponse sans lien clair avec le contexte.")

    # Final Metrics
    avg_latency = sum(latencies) / len(latencies)
    relevance_accuracy = relevance_hits / len(queries)
    context_match_rate = context_hits / len(queries)

    print("\n📊 Résultats finaux :")
    print(f"⏳ Latence moyenne : {round(avg_latency, 2)} secondes")
    print(f"🎯 Précision de pertinence : {round(relevance_accuracy * 100, 1)}%")
    print(f"📚 Taux de correspondance au contexte : {round(context_match_rate * 100, 1)}%")

# === Example Test Cases ===
if __name__ == "__main__":
    test_queries = [
        "Comment configurer l'horloge du STM32 ?",
        "Comment utiliser le timer pour générer une interruption ?",
        "Quelle est la différence entre NVIC et EXTI ?"
    ]

    reference_answers = [
        "Pour configurer l’horloge du STM32, vous devez configurer le RCC et le PLL.",
        "Le timer génère une interruption en configurant le registre DIER et en activant l’IRQ correspondante dans le NVIC.",
        "NVIC gère les priorités d’interruption, EXTI déclenche les interruptions externes basées sur les entrées GPIO."
    ]

    evaluate_all(test_queries, reference_answers)
