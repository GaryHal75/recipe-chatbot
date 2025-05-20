import sqlite3
import json
import re
import openai
from dotenv import load_dotenv
import os
import tiktoken
from datetime import datetime

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

# Correct OpenAI Client initialization for Python 3.13+
client = openai.Client(api_key=API_KEY)

# Database path (Adjust if needed)
DB_PATH = "recipe_text_chunks.db"

# JSON output path
JSON_OUTPUT_PATH = "recipe_map.json"

# Function to clean up text before processing
def clean_text(text):
    """
    Cleans up the recipe text by removing unnecessary elements while preserving recipe names.
    - Removes boilerplate headers, emails, URLs, phone numbers, and excess whitespace.
    """
    text = re.sub(r"(?i)FRANCHISE DISCLOSURE DOCUMENT", "", text)
    text = re.sub(r"(?i)SECTION HEADING", "", text)
    text = re.sub(r"\b(?:www\.|http)\S+\b", "", text)  # Remove URLs
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)  # Remove emails
    text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", "", text)  # Remove standard phone numbers
    text = re.sub(r"\b\d{8,}\b", "", text)  # Remove large document numbers
    return text.strip()

# Function to fetch relevant text chunks from database for a specific filename
def fetch_recipe_chunks(filename):
    """
    Retrieves recipe text chunks and their chunk ranges where is_deleted = 0 for a specific filename.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT id, content, chunk_index
        FROM vector_chunks 
        WHERE is_deleted = 0 AND filename = ?
    """
    
    cursor.execute(query, (filename,))
    results = cursor.fetchall()
    conn.close()
    
    return results  # Returns a list of chunks

# Function to extract recipe names using ChatGPT
def extract_recipe_names(text_chunk):
    """
    Uses OpenAI's ChatGPT to extract recipe names from the given text chunk.
    - Ensures structured output in a list format.
    - Handles token constraints by limiting input size.
    """
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    tokens = encoding.encode(text_chunk)

    # Split text_chunk if it exceeds token limit
    max_tokens = 4096  # Adjust based on model
    if len(tokens) > max_tokens:
        text_chunks = [text_chunk[i:i + max_tokens] for i in range(0, len(text_chunk), max_tokens)]
        print(f"📊 Token count for chunk split: {len(text_chunks)} chunks generated.")
    else:
        text_chunks = [text_chunk]

    extracted_names = set()
    for i, chunk in enumerate(text_chunks):
        print(f"📊 Token count for chunk {i + 1}: {len(encoding.encode(chunk))}")
        prompt = f"""
        Extract all recipe names from the following document text.
        - Return ONLY recipe names.
        - Do NOT include any extra descriptors.
        - Ignore addresses, phone numbers, and website links.
        - Output each recipe name on a separate line.

        **Text:**
        ---
        {chunk}
        ---

        **Output only the recipe names as a numbered list.**
        """

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that extracts recipe names from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500  # Start small; adjust if needed
        )

        extracted_text = response.choices[0].message.content.strip()
        names = {line.split(". ", 1)[-1].strip() for line in extracted_text.split("\n") if line.strip()}
        extracted_names.update(names)  # Add to set to ensure uniqueness
    
    return sorted(extracted_names)  # Returns sorted, deduplicated recipe names

# Main function to process recipe documents and store recipe names in JSON
def process_recipe_names():
    """
    Processes recipe names from text chunks, cleans the text, and generates recipe_map.json.
    """
    recipe_map = {}

    # Fetch unique filenames from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT filename FROM vector_chunks WHERE is_deleted = 0")
    filenames = cursor.fetchall()
    conn.close()

    for (filename,) in filenames:
        chunks = fetch_recipe_chunks(filename)
        print(f"🔍 Processing file: {filename} ({len(chunks)} chunks)")

        issuance_date_match = None
        for row_id, content, chunk_index in chunks:
            if chunk_index == 0:
                issuance_date_match = re.search(r"[Ii][Ss]{2}[Uu][Aa][Nn][Cc][Ee]\s*[Dd][Aa][Tt][Ee]\s*[:\-]?\s*(?:[A-Za-z]+\s+\d{1,2},\s+\d{4})", content)
                break
        if issuance_date_match:
            year_match = re.search(r"\d{4}", issuance_date_match.group())
            year = int(year_match.group()) if year_match else 0
        else:
            year = 0

        # Pass 1: Extract the primary recipe name (only chunk_index = 0)
        primary_chunk = next((content for row_id, content, chunk_index in chunks if chunk_index == 0), None)
        if primary_chunk:
            cleaned_primary_text = clean_text(primary_chunk)
            primary_recipe_names = extract_recipe_names(cleaned_primary_text)
            primary_recipe_name = primary_recipe_names[0] if primary_recipe_names else None

        # Pass 2: Process the rest of the chunks for JSON mapping
        for row_id, content, chunk_index in chunks:
            cleaned_text = clean_text(content)
            recipe_names = [primary_recipe_name] if primary_recipe_name else extract_recipe_names(cleaned_text)

            for name in recipe_names:
                recipe_key = f"{name}_{year}"
                print(f"✍ Extracted recipe: {name} ({recipe_key})")
                if recipe_key not in recipe_map:
                    recipe_map[recipe_key] = {
                        "name": name,
                        "year": year,
                        "filename": filename,
                        "starting_row_id": row_id,
                        "last_row_id": row_id,
                        "related_doc_row_ids": [],
                        "created_at": datetime.now().strftime("%Y-%m-%d")
                    }
                if row_id not in recipe_map[recipe_key]["related_doc_row_ids"]:
                    recipe_map[recipe_key]["related_doc_row_ids"].append(row_id)
                    recipe_map[recipe_key]["last_row_id"] = max(recipe_map[recipe_key]["last_row_id"], row_id)

    # Write the extracted recipe names to a JSON file
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(recipe_map, f, indent=4)

    print(f"✅ Recipe extraction complete! {len(recipe_map)} recipes saved to {JSON_OUTPUT_PATH}")

# Run the process when script is executed
if __name__ == "__main__":
    process_recipe_names()
