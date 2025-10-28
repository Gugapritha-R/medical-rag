import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# ⚙️ Setup
# -------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")
INDEX_NAME = "medical-rag"

# -------------------------------
# 🔧 Initialize Clients
# -------------------------------
app = FastAPI(title="Medical RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# -------------------------------
# 📥 Request / 📤 Response Models
# -------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    contexts: list[str]

# -------------------------------
# 🔍 Context Retrieval
# -------------------------------
def retrieve_context(query: str, top_k: int = 3):
    query_embedding = embedder.encode(query).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="medical"
    )

    if not results.matches:
        return []

    contexts = [m.metadata.get("text", "") for m in results.matches if m.metadata]
    return contexts

# -------------------------------
# 💬 Query Groq LLM
# -------------------------------
def query_groq_llm(query: str, contexts: list[str]):
    if not contexts:
        return {"answer": "⚠️ No relevant context found.", "contexts": []}

    combined_context = "\n\n---\n\n".join(contexts)
    prompt = f"Use the following context to answer the question concisely and accurately:\n\n{combined_context}\n\nQuestion: {query}"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a medical assistant that answers strictly from the given context."},
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
        raise HTTPException(status_code=response.status_code, detail="Groq LLM error")

# -------------------------------
# 🧠 POST /query Endpoint
# -------------------------------
@app.post("/query", response_model=QueryResponse)
def rag_query(body: QueryRequest):
    try:
        contexts = retrieve_context(body.query, top_k=body.top_k)
        result = query_groq_llm(body.query, contexts)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 🏠 Root Route
# -------------------------------
@app.get("/")
def root():
    return {"message": "🩺 Medical RAG API is running!"}

# -------------------------------
# 🚀 Run (for local)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🚀 Medical RAG API running at http://127.0.0.1:8000/query")
    uvicorn.run(app, host="0.0.0.0", port=8000)
