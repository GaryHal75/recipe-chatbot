import sqlite3

DB_PATH = "recipe_text_chunks.db"

def setup_vector_chunks_table():
    """Creates the vector_chunks table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vector_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            model TEXT DEFAULT 'openai-ada-002',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("✅ vector_chunks table is ready.")

if __name__ == "__main__":
    setup_vector_chunks_table()