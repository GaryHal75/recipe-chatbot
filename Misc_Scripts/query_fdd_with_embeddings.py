import sqlite3
import openai
import os
import json
import numpy as np
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

client = openai.OpenAI(api_key=API_KEY)

DB_PATH = "fdd_text_chunks.db"
TOP_K = 5  # Number of top matches to retrieve

def get_embedding(text):
    """Generates an embedding for the input query."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_relevant_chunks(query, filename=None):
    """Finds the most relevant stored embeddings for the given query using cosine similarity."""
    query_embedding = get_embedding(query)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if filename:
        cursor.execute("SELECT id, filename, chunk_index, content, embedding FROM vector_chunks WHERE filename = ?", (filename,))
    else:
        cursor.execute("SELECT id, filename, chunk_index, content, embedding FROM vector_chunks")

    results = []
    for row in cursor.fetchall():
        stored_embedding = json.loads(row[4])  # Convert stored JSON string to list
        similarity = cosine_similarity(query_embedding, stored_embedding)
        results.append((row[0], row[1], row[2], row[3], similarity))

    conn.close()

    # Sort results by similarity score, descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    return results[:TOP_K]  # Return top K matches

def ask_gpt_about_fdd(query, filename=None):
    """Retrieves the best matches and sends them to GPT for structured analysis."""
    relevant_chunks = search_relevant_chunks(query, filename)

    if not relevant_chunks:
        return "No relevant information found in stored embeddings."

    compiled_text = "\n\n".join([chunk[3] for chunk in relevant_chunks])  # Extract content from results

    prompt = f"""
    You are analyzing a Franchise Disclosure Document (FDD). Extract the most relevant details based on the query:

    Query: {query}

    Relevant document sections:
    {compiled_text}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in Franchise Disclosure Documents (FDDs). Extract structured data based on the provided sections."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

if __name__ == "__main__":
    filename = input("\n📄 Enter the FDD filename (or press Enter to search all): ").strip()
    query = input("\n🔎 Enter your query about the FDD (or type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        exit()

    print("\n🚀 Searching relevant embeddings and querying GPT...")
    gpt_response = ask_gpt_about_fdd(query, filename if filename else None)

    print("\n✅ AI Response:\n")
    print(gpt_response)