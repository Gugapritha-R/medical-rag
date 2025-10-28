import json
import os
import random

class RAGModel:
    def __init__(self, processed_data_path="data/processed"):
        self.docs = []
        for file in os.listdir(processed_data_path):
            if file.endswith(".json"):
                with open(os.path.join(processed_data_path, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.docs.extend(data.get("pages", []))  # assuming each json has "pages"

    def query(self, user_query):
        if not self.docs:
            return "No documents loaded."
        # just randomly select a page as dummy RAG result
        random_page = random.choice(self.docs)
        return f"Answer (dummy RAG): Based on the book content, here's something related —\n\n{random_page[:800]}"
