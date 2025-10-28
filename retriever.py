import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests

load_dotenv()

# -------------------------------
# CONFIGURATION
# -------------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
INDEX_NAME = "medical-rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = os.environ.get("MODEL", "llama-3.3-70b-versatile")

# -------------------------------
# INITIALIZE CLIENTS
# -------------------------------
embedder = SentenceTransformer(EMBED_MODEL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# RETRIEVAL FUNCTION
# -------------------------------
def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="medical"
    )

    if not results.matches:
        return None

    contexts = [m.metadata.get("text", "") for m in results.matches if m.metadata]
    return contexts  # list of snippets

# -------------------------------
# ANSWER GENERATION FUNCTION (Groq)
# -------------------------------
def generate_answer(query, contexts):
    if not contexts:
        return {"answer": "⚠️ No relevant context found.", "contexts": []}

    combined_context = "\n\n---\n\n".join(contexts)
    prompt = f"Use the following context to answer the question concisely and factually:\n\n{combined_context}\n\nQuestion: {query}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a medical assistant that answers only from the given context."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "contexts": contexts}
    else:
        print("LLM Error:", response.text)
        return {"answer": f"⚠️ Error: {response.status_code}", "contexts": contexts}

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    query = input("🧠 Query: ")

    try:
        contexts = retrieve_context(query)
        if contexts:
            print("\n📚 Retrieved Context (first 400 chars):\n", contexts[0][:400], "...\n")
        else:
            print("⚠️ No context retrieved.")

        result = generate_answer(query, contexts)
        print("\n✅ Final Answer:\n", result["answer"])
        print("\n📖 Cited Contexts:")
        for i, ctx in enumerate(result["contexts"], 1):
            print(f"  {i}. {ctx[:200]}...")

    except Exception as e:
        print("❌ Error:", str(e))
