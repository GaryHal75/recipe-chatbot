import sqlite3
import faiss
import numpy as np
import json
import os
import re

#IMPORTANT: run this command in terminal to create the FAISS index
#/SlidingWindow/venv/bin/python -c "import faiss_index_4; faiss_index_4.build_and_save_index()"

# Create FAISS Index from Vector Chunks to prep for Search FAISS

DB_PATH = "recipe_text_chunks.db"
FAISS_INDEX_FILE = "faiss_index.idx"  # Where the FAISS index is stored

def load_embeddings():
    """Loads embeddings from SQLite for FAISS indexing."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, chunk_index, embedding
        FROM recipe_embeddings
        WHERE is_embedded = 1 AND is_deleted = 0 AND model = ?
    """, ("text-embedding-ada-002",))
    
    ids, metadata, embeddings = [], [], []
    
    for row in cursor.fetchall():
        idx, filename, chunk_index, embedding = row
        vector = np.array(json.loads(embedding), dtype=np.float32)  # Convert JSON to numpy array

        ids.append(idx)
        metadata.append((filename, chunk_index))
        embeddings.append(vector)
    
    print(f"📊 Loaded {len(embeddings)} embeddings from the database.")
    assert all(vec.shape[0] == embeddings[0].shape[0] for vec in embeddings), "Inconsistent embedding dimensions!"
    conn.close()
    return np.array(embeddings, dtype=np.float32), ids, metadata

def build_and_save_index():
    """Builds FAISS index and saves it to disk with correct SQLite row mappings."""
    embeddings, ids, metadata = load_embeddings()

    if embeddings.shape[0] == 0:
        print("❌ No embeddings loaded. Skipping FAISS index creation.")
        return

    # Create a FAISS index (L2 distance)
    index = faiss.IndexFlatL2(embeddings.shape[1])

    # Convert IDs to numpy array (FAISS requires 32-bit integers)
    id_map = np.array(ids, dtype=np.int32)

    # FAISS index should store row mappings
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, id_map)

    # Debug log just before writing the index
    print(f"💾 Preparing to write FAISS index with {len(ids)} vectors to {FAISS_INDEX_FILE}")
    # Save the index to disk with error handling
    try:
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"✅ FAISS index saved with {len(ids)} vectors, mapped to SQLite row IDs.")
    except Exception as e:
        print(f"❌ Failed to write FAISS index: {e}")

def load_faiss_index():
    """Loads FAISS index from disk, or rebuilds it if missing."""
    if os.path.exists(FAISS_INDEX_FILE):
        print(f"🔄 Loading FAISS index from {FAISS_INDEX_FILE}...")
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS index loaded with {index.ntotal} vectors.")
    else:
        print("⚠️ FAISS index not found. Rebuilding...")
        build_and_save_index()
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS index rebuilt with {index.ntotal} vectors.")
    
    return index

def search_faiss(query_embedding, top_k=5):
    """Finds the most relevant text chunks using FAISS and retrieves correct content."""
    index = load_faiss_index()
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)

    print(f"🔍 Searching FAISS returned indices (Mapped IDs as row_id): {indices[0]}")  # Debugging

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    results = []
    for i in range(top_k):
        row_id = int(indices[0][i])  # Correctly storing as row_id
        if row_id < 0:  # FAISS may return -1 if no matches
            continue

        # Debug: Print the ID FAISS is trying to fetch
        print(f"🧐 Fetching text for SQLite row ID: {row_id}")

        # Query updated to fetch using the correct ID column
        cursor.execute("""
            SELECT id, filename, chunk_index, content
            FROM recipe_embeddings
            WHERE id = ?
            """, (row_id,))  # Only using id, not chunk_index

        row = cursor.fetchone()

        if row:
            results.append({
                "row_id": row[0],  # Explicitly storing row_id
                "chunk_index": row[2],
                "filename": row[1],
                "text": row[3],
                "distance": float(distances[0][i])
            })
        else:
            print(f"⚠️ No matching text found for SQLite row ID {row_id}.")

    conn.close()
    return results

if __name__ == "__main__":
    build_and_save_index()