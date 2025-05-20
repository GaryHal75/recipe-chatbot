import openai
from dotenv import load_dotenv
import numpy as np
import os
import re
import json
from faiss_index_4 import search_faiss

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=API_KEY)

def generate_query_embedding(query):
    """Creates an embedding for the user's query."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

def extract_top_k(query):
    match = re.search(r'\b(\d+)\b', query)
    return int(match.group(1)) if match else 3  # Default to 3 if no number found

def search_and_filter(query, row_id_scope=None):
    print("🔍 Checking FAISS for best matches...")
    query_embedding = generate_query_embedding(query)
    requested_top_k = extract_top_k(query)

    results = search_faiss(query_embedding, top_k=requested_top_k + 5)
    print("✅ Processing results...")

    grouped = {}
    for r in results:
        filename = r.get("filename", "[Unknown]")
        if filename not in grouped:
            grouped[filename] = {
                "filename": filename,
                "chunks": [],
                "best_score": float("inf")
            }

        chunk_data = {
            "row_id": r["row_id"],
            "chunk_index": r["chunk_index"],
            "distance_score": round(r["distance"], 4),
            "text": r.get("text", "[No content found]")
        }

        grouped[filename]["chunks"].append(chunk_data)
        if r["distance"] < grouped[filename]["best_score"]:
            grouped[filename]["best_score"] = r["distance"]

    # Sort by best score across recipes
    structured_results = sorted(
        grouped.values(),
        key=lambda x: x["best_score"]
    )

    return structured_results

if __name__ == "__main__":
    query = input("Enter search query: ")
    results = search_and_filter(query)
    print("\n🔍 **Search Results:**")
    if not results or "error" in results[0]:
        print(f"⚠️ {results[0].get('error', 'No search results found.')}")
    else:
        for item in results:
            print(f"\n📄 {item['filename']} | Best Score: {round(item['best_score'],4)}")
            for chunk in item['chunks']:
                print(f"  Chunk {chunk['chunk_index']} | Score: {chunk['distance_score']}")
                print(f"  {chunk['text']}")