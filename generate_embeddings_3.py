import sqlite3
import openai
import os
import json
import tiktoken  # Tokenizer to count tokens
import hashlib
from dotenv import load_dotenv

#Step 3: Generating Embeddings from Text Chunks to create Vector Chunks for the vector_chunks table. 

# Load API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

client = openai.OpenAI(api_key=API_KEY)

DB_PATH = "recipe_text_chunks.db"
MAX_TOKENS = 2000  # Reduce per-chunk size to minimize memory overload
OVERLAP_TOKENS = 100  # Overlapping tokens for continuity

def count_tokens(text):
    """Returns the number of tokens in a given text using OpenAI's tokenizer."""
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    return len(enc.encode(text))

def split_large_text(text, max_tokens=1024, overlap=100):
    """Splits text into chunks with reduced overlap to prevent duplication."""
    enc = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = enc.encode(text)
    start = 0
    sub_index = 0
    seen_hashes = set()

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        # Generate hash for uniqueness check
        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
        if chunk_hash not in seen_hashes:
            yield sub_index, chunk_text
            seen_hashes.add(chunk_hash)

        sub_index += 1
        start += (max_tokens - overlap)  # Reduce overlap progression

def fetch_text_chunks():
    """Fetches chunks one at a time from the database to prevent memory overload."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, chunk_index, content FROM text_chunks ORDER BY filename, chunk_index ASC")

    while True:
        row = cursor.fetchone()
        if not row:
            break
        yield row  # Yields only one row at a time instead of loading all into memory

    conn.close()

def store_embedding(chunk_id, filename, chunk_index, content, embedding, model="text-embedding-ada-002"):
    """Stores a single embedding in SQLite, ensuring no duplicates and avoiding lock issues."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()

        # Ensure WAL mode is enabled (Write-Ahead Logging)
        cursor.execute("PRAGMA journal_mode=WAL;")

        cursor.execute(
            "INSERT INTO vector_chunks (chunk_id, filename, chunk_index, content, embedding, model) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, filename, chunk_index, content, json.dumps(embedding), model)
        )
        conn.commit()

    except sqlite3.IntegrityError:
        print(f"⚠️ Skipping duplicate chunk {chunk_index} for {filename}")
    except sqlite3.OperationalError as e:
        print(f"⚠️ Database error: {e}")
    finally:
        if conn:
            conn.close()  # Ensure connection is always closed

def generate_and_store_embeddings():
    """Processes text chunks sequentially, preventing RAM overload and duplicates."""
    seen_chunks = set()  # Track unique chunks

    for chunk_id, filename, chunk_index, content in fetch_text_chunks():
        token_count = count_tokens(content)

        if token_count > MAX_TOKENS:
            print(f"⚠️ Chunk {chunk_index} is too large ({token_count} tokens). Splitting...")
            for sub_index, sub_chunk in split_large_text(content):
                indexed_chunk = f"{chunk_index}.{sub_index}"

                # Ensure we aren't embedding duplicate chunks
                chunk_hash = hashlib.md5(sub_chunk.encode()).hexdigest()
                if chunk_hash in seen_chunks:
                    print(f"⚠️ Skipping duplicate chunk {indexed_chunk} for {filename}")
                    continue
                seen_chunks.add(chunk_hash)

                print(f"🔄 Generating embedding for {filename} - Chunk {indexed_chunk}...")
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=sub_chunk
                )
                embedding = response.data[0].embedding
                store_embedding(chunk_id, filename, indexed_chunk, sub_chunk, embedding)

        else:
            # Ensure we aren't embedding duplicate chunks
            chunk_hash = hashlib.md5(content.encode()).hexdigest()
            if chunk_hash in seen_chunks:
                print(f"⚠️ Skipping duplicate chunk {chunk_index} for {filename}")
                continue
            seen_chunks.add(chunk_hash)

            print(f"🔄 Generating embedding for {filename} - Chunk {chunk_index}...")
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=content
            )
            embedding = response.data[0].embedding
            store_embedding(chunk_id, filename, chunk_index, content, embedding)

    print("✅ All embeddings have been stored successfully.")

if __name__ == "__main__":
    generate_and_store_embeddings()