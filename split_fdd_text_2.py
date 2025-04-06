import os
import sqlite3
import tiktoken

#Step 2: Splitting the FDD Text into Chunks

# Configurations
DB_PATH = "recipe_text_chunks.db"
INPUT_FOLDER = "Outputs/"  # Process all cleaned text files
TOKEN_LIMIT = 10000  # Max tokens per chunk
OVERLAP = 1500  # Tokens that overlap between chunks

def count_tokens(text):
    """Returns the token count for a given text."""
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    return len(enc.encode(text))

def split_text_into_windows(text):
    """Splits the full document text into overlapping sliding windows."""
    words = text.split()  # Basic tokenization
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + TOKEN_LIMIT, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move start forward, keeping an overlap
        start = end - OVERLAP if end < len(words) else len(words)

    return chunks

def store_chunks_in_db(filename, chunks):
    """Stores generated text chunks in SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)
        cursor.execute("INSERT INTO text_chunks (filename, chunk_index, content, token_count, doc_type, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                       (filename, i, chunk, token_count, "general", None))

    conn.commit()
    conn.close()
    print(f"✅ Stored {len(chunks)} text chunks from {filename} in the database.")

def process_fdd_text():
    """Reads all cleaned FDD text files, splits them into chunks, and stores them in the database."""
    if not os.path.exists(INPUT_FOLDER):
        print(f"🚨 Error: `{INPUT_FOLDER}` folder not found. Ensure text files exist.")
        return

    text_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]

    if not text_files:
        print(f"🚨 Error: No `.txt` files found in `{INPUT_FOLDER}`.")
        return

    for file in text_files:
        file_path = os.path.join(INPUT_FOLDER, file)
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = split_text_into_windows(full_text)
        store_chunks_in_db(file, chunks)
        print(f"✅ Processed and stored chunks from {file}.")

if __name__ == "__main__":
    process_fdd_text()