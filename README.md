# 🧠 SlidingWindow Recipe Chatbot

A full-stack chatbot powered by OpenAI embeddings + FAISS, designed to search, compare, and remix recipes extracted from structured PDF files.

## 🔍 Key Features

- PDF ingestion, text chunking, and vector embedding with OpenAI
- FAISS index for fast semantic search across recipes
- Structured chatbot interface using Flask and session memory
- Multi-recipe comparison and ingredient analysis
- Resettable conversation flow with friendly welcome state
- Future-ready for creative recipe synthesis
- Structured FAISS search via `search_faiss_5.py` with grouped recipe results for contextual richness

## ⚙️ Local Setup

1. **Clone this repo**
2. **Create your `.env`** from `.env.example`
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the full pipeline:**
   ```bash
   python batch_pdf_to_text_1.py          # Extract PDF content and convert to text and json format
   python setup_text_db.py                # Create Database for embeddings
   python split_recipe_text_2.py          # Extract content from text to create unique table rows
   python generate_embeddings_3.py        # Create embeddings from unique rows
   python faiss_index_4.py                # Build FAISS index from database
   python chatbot.py                      # Launch web app
   ```

## 💬 Web Interface

- Runs at: `http://localhost:5001`
- Type queries like:
  - `find all recipes with beans`
  - `which recipes use carrots?`
  - `combine ingredients from bean recipes into a new one`

## 📁 Project Structure

```
.
├── Inputs/              # PDF recipes
├── Outputs/             # Flattened + structured text chunks
├── faiss_index_4.py     # FAISS index build + search logic
├── generate_embeddings_3.py
├── search_faiss_5.py              # Grouped semantic search interface
├── chatbot.py           # Flask app & GPT interface
├── templates/           # HTML front-end
└── recipe_text_chunks.db  # SQLite storage
```

## ✨ Author

Built by [Gary Blanchard](https://github.com/GaryHal75) — creative dev focused on AI, automation, and meaningful tools.
