import sqlite3
import os

# Create text chunk database to store text chunks extracted from recipe PDFs.

# Database path
DB_PATH = "recipe_text_chunks.db"

def setup_text_database():
    """Creates a SQLite database for storing text chunks extracted from PDFs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop the old text_chunks table if it exists to avoid conflicts
    cursor.execute('DROP TABLE IF EXISTS text_chunks')

    # Drop the recipe_embeddings table if it exists, then create the enhanced schema
    cursor.execute('DROP TABLE IF EXISTS recipe_embeddings')

    cursor.execute('''
        CREATE TABLE recipe_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            embedding TEXT,
            model TEXT DEFAULT 'openai-ada-002',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            is_embedded INTEGER DEFAULT 0,
            is_deleted INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database `{DB_PATH}` has been created and initialized.")

if __name__ == "__main__":
    setup_text_database()