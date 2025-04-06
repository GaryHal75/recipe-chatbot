import sqlite3
import faiss
import numpy as np
import ast
from transformers import pipeline

DB_PATH = "fdd_text_chunks.db"
FAISS_INDEX_PATH = "faiss_index.idx"

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# Load Named Entity Recognition (NER) model from Transformers
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def extract_franchise_list():
    """Extracts franchise names using FAISS & NER, focusing on chunk_index = 0."""
    
    cursor.execute("SELECT filename, embedding, content FROM vector_chunks WHERE chunk_index = 0")
    chunks = cursor.fetchall()

    if not chunks:
        print("⚠️ No chunks found! Verify the database contains data.")
        return []

    franchise_names = set()

    for filename, vector_blob, text_chunk in chunks:
        try:
            print(f"\n🔹 Processing File: {filename}")

            # Convert embedding from stored JSON string
            vector_list = ast.literal_eval(vector_blob)
            if not isinstance(vector_list, list):
                print(f"⚠️ Skipping {filename}: Embedding is not a valid list")
                continue

            vector = np.array(vector_list, dtype=np.float32)

            print(f"🔹 FAISS Expected Dim: {index.d}, Vector Shape: {vector.shape}")
            # Ensure correct vector shape
            if vector.shape[0] != index.d:
                print(f"⚠️ Skipping {filename}: Vector shape mismatch (Expected: {index.d}, Found: {vector.shape[0]})")
                continue

            # Search FAISS for the most relevant match
            try:
                distances, indices = index.search(np.expand_dims(vector, axis=0), 1)
            except Exception as e:
                print(f"❌ FAISS Search Error for {filename}: {e}")
                continue

            best_match_id = indices[0][0]
            best_match_distance = distances[0][0]
            print(f"🔎 FAISS Best Match: ID={best_match_id}, Distance={best_match_distance}")

            print(f"🔎 FAISS Best Match ID Type: {type(best_match_id)}, Value: {best_match_id}")
            if best_match_id == -1:
                print(f"⚠️ No valid FAISS match for {filename}")
                continue

            print(f"🔎 FAISS Search Result for {filename}: Best Match ID = {best_match_id}")

            # Check for valid embedding data
            cursor.execute("SELECT id, filename, LENGTH(embedding) FROM vector_chunks WHERE id = ?", (int(best_match_id),))
            embedding_check = cursor.fetchone()
            if embedding_check and embedding_check[2] == 0:
                print(f"⚠️ Warning: No valid embedding found for ID {best_match_id}. Skipping {filename}.")
                continue

            # Fetch corresponding text chunk
            print(f"🛠️ Debug: Querying DB with ID: {best_match_id}")
            cursor.execute("SELECT filename, content FROM vector_chunks WHERE id = ?", (int(best_match_id),))
            result = cursor.fetchone()
            print(f"🛠️ Debug: Fetched DB result: {result}")
            print(f"🛠️ Debug: Retrieved from DB - Filename: {result[0] if result else 'None'}, Content Length: {len(result[1]) if result else 'None'}")
            print(f"\n🔎 FAISS Matched Text for {filename}: {result}\n{'='*80}")

            if result:
                extracted_text = result[1].strip()
                
                if not extracted_text:
                    print(f"⚠️ No text extracted from FAISS for {filename}.")
                    continue

                print(f"📝 Extracted Text Chunk for {filename} ({len(extracted_text)} characters):\n{extracted_text}\n{'='*80}")
                
                print(f"🔍 Running NER on extracted text for {filename}...")
                ner_results = ner_pipeline(extracted_text)
                print(f"🧠 Raw NER Output for {filename}: {ner_results}")

                # Extract only organization names (I-ORG)
                entities = []
                current_entity = ""

                for entry in ner_results:
                    print(f"🔹 Processing entity: {entry}")  # Debug log
                    word = entry["word"]

                    if entry["entity"].startswith("I-ORG"):
                        if word.startswith("##"):
                            current_entity += word[2:]  # Append subword
                        else:
                            if current_entity:
                                entities.append(current_entity)  # Store completed entity
                                print(f"✅ Completed Entity: {current_entity}")  
                            current_entity = word
                    else:
                        if current_entity:
                            entities.append(current_entity)
                            print(f"✅ Finalized Entity: {current_entity}")
                            current_entity = ""

                if current_entity:
                    entities.append(current_entity)

                print(f"🔍 Extracted Entities from NER (Raw): {entities}")

                # Reconstruct full names from detected entities
                reconstructed_names = " ".join(entities)

                # Validation: Ensure name is long enough and meaningful
                valid_name = (
                    len(reconstructed_names) > 5 and  # Minimum length
                    any(x in reconstructed_names.lower() for x in ["franchise", "coffee", "inc", "company", "group"])
                )

                if valid_name:
                    franchise_names.add(reconstructed_names)
                    print(f"✅ Valid Franchise Name Extracted: {reconstructed_names}")
                else:
                    print(f"❌ Ignored Invalid Name: {reconstructed_names}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    # Manual NER test snippet
    test_text = "Biggby Coffee is a growing franchise in the USA."
    ner_results = ner_pipeline(test_text)
    print(f"🧠 Test NER Output: {ner_results}")

    return sorted(franchise_names)

if __name__ == "__main__":
    franchises = extract_franchise_list()
    print("\n✅ **Final Extracted Franchise Names:**")
    for name in franchises:
        print("-", name)

    print(f"\n🎯 **Total Unique Franchises Extracted: {len(franchises)}** (Expected: 29)")