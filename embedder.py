import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
DATA_DIR = os.path.join(".", "extracted_data")
INDEX_META_PATH = os.path.join(DATA_DIR, "index.json")

# Initialize the embedder (MiniLM gives 384-dim embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Improved Semantic Chunker ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # slightly bigger for medical texts
    chunk_overlap=150,     # overlap for context continuity
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

def chunk_text(text):
    """Chunk text into semantically meaningful pieces."""
    return text_splitter.split_text(text)

# --- Embedding Function ---
def process_book(book_id, file_info):
    file_path = os.path.normpath(file_info["output"])
    with open(file_path, "r", encoding="utf-8") as f:
        book_data = json.load(f)

    embeddings = []
    for page_num, text in tqdm(book_data["pages"].items(), desc=f"Processing {book_id}"):
        if not text.strip():
            continue  # skip empty pages

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            embeddings.append({
                "id": f"{book_id}_page_{page_num}_chunk_{idx}",
                "values": embedding.tolist(),
                "metadata": {
                    "book_id": book_id,
                    "page": int(page_num),
                    "chunk_index": idx,
                    "text": chunk
                }
            })
    return embeddings

# --- Main Loop ---
def main():
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        index_meta = json.load(f)

    all_embeddings = []

    for book_id, info in index_meta.items():
        print(f"\n📘 Embedding: {book_id} -> {info['file_name']}")
        book_embeddings = process_book(book_id, info)
        all_embeddings.extend(book_embeddings)

    # Save locally before uploading to Pinecone
    output_path = os.path.join(DATA_DIR, "embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_embeddings, f, indent=2)

    print(f"\n✅ Embeddings generated and saved at: {output_path}")
    print(f"Total vectors: {len(all_embeddings)}")

if __name__ == "__main__":
    main()
