import sqlite3

DB_PATH = "recipe_text_chunks.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Enable WAL mode
cursor.execute("PRAGMA journal_mode=WAL;")

# Make a dummy write to force WAL/SHM creation
cursor.execute("CREATE TABLE IF NOT EXISTS wal_trigger (id INTEGER)")
cursor.execute("INSERT INTO wal_trigger (id) VALUES (1)")

conn.commit()
conn.close()

print("✅ WAL mode activated and files should now be created.")