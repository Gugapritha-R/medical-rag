import os
import json
import glob
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# -------------------------------
# 🔧 Load Environment Variables
# -------------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

# -------------------------------
# ⚙️ Groq Configuration
# -------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")

# Directory where your processed book JSONs are stored
DATA_DIR = os.path.join("data", "processed")


# -------------------------------
# 🔍 Simple Keyword-Based Retriever
# -------------------------------
def retrieve_relevant_contexts(query, top_k=3):
    contexts = []
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                # Handle both { "pages": [...] } and [ { "text": ... } ] structures
                pages = data.get("pages", data) if isinstance(data, dict) else data

                for page in pages:
                    text = ""
                    if isinstance(page, dict):
                        text = page.get("text", "")
                    elif isinstance(page, str):
                        text = page
                    if any(word.lower() in text.lower() for word in query.split()):
                        contexts.append(text)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    contexts = sorted(contexts, key=len, reverse=True)[:top_k]
    return contexts


# -------------------------------
# 🧠 Function to Query Groq LLM
# -------------------------------
def query_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers based on the given context.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        print("LLM Error:", response.text)
        return f"Error: {response.status_code}"


# -------------------------------
# 💬 RAG Endpoint
# -------------------------------
@app.route("/query", methods=["POST"])
def rag_query():
    try:
        user_query = request.json.get("query", "")
        top_k = int(request.json.get("top_k", 3))

        # Retrieve contexts
        contexts = retrieve_relevant_contexts(user_query, top_k=top_k)

        # Combine query and context
        if contexts:
            combined_prompt = (
                "Use the following context to answer the question accurately:\n\n"
                + "\n\n---\n\n".join(contexts)
                + f"\n\nQuestion: {user_query}"
            )
        else:
            combined_prompt = f"Question: {user_query}"

        # Get LLM response
        answer = query_llm(combined_prompt)

        return jsonify({"query": user_query, "contexts": contexts, "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# 🌐 Frontend Route
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# 🚀 Run the App
# -------------------------------
if __name__ == "__main__":
    print("🚀 RAG API running at http://127.0.0.1:5000")
    app.run(debug=True)
