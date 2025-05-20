import os
import sqlite3
import tiktoken

# Step 2: Splitting the recipe text into chunks

# Configurations
DB_PATH = "recipe_text_chunks.db"
INPUT_FOLDER = "Outputs/flattened/"  # Process all cleaned text files
TOKEN_LIMIT = 10000  # Max tokens per chunk
OVERLAP = 1500  # Tokens that overlap between chunks

def count_tokens(text):
    """Returns the token count for a given text."""
    enc = tiktoken.encoding_for_model("gpt-4-turbo")
    return len(enc.encode(text))

def extract_labeled_chunks(text):
    labels = ["TITLE:", "SERVINGS:", "COST:", "INGREDIENTS:", "DIRECTIONS:", "NUTRITION:", "FOOD GROUPS:", "SOURCE:"]
    chunks = []
    current_label = None
    buffer = []

    for line in text.splitlines():
        if any(line.startswith(label) for label in labels):
            if current_label and buffer:
                chunks.append(f"{current_label}\n" + "\n".join(buffer).strip())
            current_label = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if current_label and buffer:
        chunks.append(f"{current_label}\n" + "\n".join(buffer).strip())

    return chunks

def store_chunks_in_db(filename, chunks):
    """Stores generated text chunks in SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)
        cursor.execute("""
            INSERT INTO recipe_embeddings (filename, chunk_index, content, token_count, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, i, chunk, token_count, None))

    conn.commit()
    conn.close()
    print(f"✅ Stored {len(chunks)} text chunks from {filename} in the database.")

def process_recipe_text():
    """Reads all cleaned recipe text files, splits them into chunks, and stores them in the database."""
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

        chunks = extract_labeled_chunks(full_text)
        store_chunks_in_db(file, chunks)
        print(f"✅ Processed and stored chunks from {file}.")

if __name__ == "__main__":
    process_recipe_text()
