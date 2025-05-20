import sqlite3
import os

# Create text chunk database to store text chunks extracted from recipe PDFs.

# Database path
DB_PATH = "recipe_text_chunks.db"

def setup_text_database():
    """Creates a SQLite database for storing text chunks extracted from PDFs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            doc_type TEXT DEFAULT 'general',
            metadata TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database `{DB_PATH}` has been created and initialized.")

if __name__ == "__main__":
    setup_text_database()