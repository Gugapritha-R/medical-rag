import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

# --- Load environment variables ---
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY2")
INDEX_NAME = os.getenv("INDEX_NAME_2")
PINECONE_REGION = os.getenv("PINECONE_REGION")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")

DATA_DIR = os.path.join(".", "extracted_data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.json")

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Create index if missing ---
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"🔧 Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(INDEX_NAME)

# --- Load embeddings ---
if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError(f"❌ Embeddings file not found at {EMBEDDINGS_PATH}")

with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
    embeddings_data = json.load(f)

print(f"📚 Loaded {len(embeddings_data)} vectors from embeddings.json")

# --- Upload in batches ---
BATCH_SIZE = 200 # adjust based on speed vs memory
namespace = "medical"

print("📤 Uploading vectors to Pinecone...")
for i in tqdm(range(0, len(embeddings_data), BATCH_SIZE)):
    batch = embeddings_data[i : i + BATCH_SIZE]
    index.upsert(vectors=batch, namespace=namespace)

# --- Verify upload ---
stats = index.describe_index_stats()
print(f"✅ Upload completed!\n📊 Index stats: {stats}")
