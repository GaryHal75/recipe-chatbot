import sqlite3
import openai
import os
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key is missing. Ensure it is set in the .env file.")

# Correct OpenAI Client initialization for Python 3.13+
client = openai.Client(api_key=API_KEY)

DB_PATH = "fdd_text_chunks.db"

def get_first_vector_chunk():
    """Fetch the first vectorized text chunk from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT content FROM text_chunks ORDER BY chunk_index LIMIT 1;")
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None

def extract_all_franchise_names_with_chatgpt(text_chunk):
    """Force ChatGPT to extract all franchise names from the first chunk."""

    prompt = f"""
    The following text is from a Franchise Disclosure Document (FDD). 
    Your task is to extract **all** franchise brand names mentioned.

    - **There are multiple franchise names in this text. Extract them all.**
    - Do **NOT** include legal descriptors (LLC, Inc., Corp.).
    - Ignore addresses, phone numbers, and contact details.
    - Ensure every franchise name is listed separately.
    - Do **NOT** list the same franchise name twice. Remove duplicates.

    **Text:**
    ---
    {text_chunk}
    ---

    **Output ALL unique franchise names as a numbered list. Do NOT stop at just a few names.**
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that extracts all franchise names from text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=750  # Increased from 500 to 750 to capture more names
    )

    extracted_text = response.choices[0].message.content.strip()

    # Extract names from numbered list format and deduplicate
    extracted_names = {line.split(". ", 1)[-1].strip() for line in extracted_text.split("\n") if line.strip()}  # Set removes duplicates

    return sorted(extracted_names)  # Sorting for consistency

# Get the first vector chunk
vector_chunk = get_first_vector_chunk()

if vector_chunk:
    franchise_names = extract_all_franchise_names_with_chatgpt(vector_chunk)
    print("\nExtracted Franchise Names:")
    for name in franchise_names:
        print("-", name.strip())  # Clean up extra spaces
    print(f"\n🎯 Total Franchises Found: {len(franchise_names)}")
else:
    print("No vector chunks found.")