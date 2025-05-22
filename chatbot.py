from flask import Flask, request, jsonify, Response, render_template, session
import openai
import os
import time
import json
import re
from dotenv import load_dotenv
from search_faiss_5 import search_and_filter
from flask_session import Session
import sqlite3

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("CHAT_SESSION_KEY")

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./.flask_session"
app.config["SESSION_PERMANENT"] = False  # avoids stale data from old sessions
Session(app)

def is_new_topic(query, last_topic):
    lowered_query = query.lower()
    if "__RESET_CHAT__" in lowered_query:
        return True
    reset_keywords = ["clear everything", "reset from scratch", "reset conversation"]
    return any(keyword in lowered_query for keyword in reset_keywords)

def is_followup_query(query):
    followup_keywords = ["compare", "which one", "these", "those", "the second", "that one", "how about", "what about"]
    return any(kw in query.lower() for kw in followup_keywords)

@app.route("/")
def home():
    session.clear()
    session["token_usage"] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0
    }
    return render_template("index.html")

def stream_gpt_response(user_query, context_text, chat_history):
    MAX_CHAT_HISTORY = 6
    chat_history = chat_history[-MAX_CHAT_HISTORY:]
    messages = [{
        "role": "system",
        "content": (
            "You are a clear, structured, and helpful assistant. Always format your responses using readable HTML for display in a web browser.\n\n"
            "📄 Format Rules:\n"
            "- Wrap all paragraphs in <p> tags and add space between them.\n"
            "- Use <ul> and <li> for lists (ingredients, steps, comparisons).\n"
            "- Use <strong> for highlighting key values or important points.\n"
            "- Separate sections with clear headings if multiple categories exist (e.g., Ingredients, Instructions, Nutrition).\n"
            "- Keep responses concise, avoiding repetition.\n"
            "- Do not mention that the context came from chunks; treat the information as native knowledge.\n\n"
            "💡 Examples:\n"
            "<p>This recipe is low in sodium and takes only 30 minutes to prepare.</p>\n"
            "<ul>\n  <li><strong>Calories:</strong> 250</li>\n  <li><strong>Fat:</strong> 5g</li>\n</ul>\n"
            "<p>To make this dish, start by preheating the oven to 375°F...</p>\n\n"
            "🎯 Use the structured context chunks provided to answer the user's question.\n"
            "If the information is not available, say so clearly and politely."
        )
    }]
    for msg in chat_history:
        if msg["content"] != "Generating response...":
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"{user_query}\n\n### Relevant Data:\n{context_text}"
    })

    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4o")
    input_token_count = sum(len(encoding.encode(m["content"])) for m in messages)
    print(f"🧮 Total token count into GPT: {input_token_count}")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True
    )

    def generate():
        output_token_count = 0
        full_text = ""
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_text += text
                output_token_count += len(encoding.encode(text))
                yield text

        # Update session usage outside of response streaming
        from flask import has_request_context
        if has_request_context():
            if "token_usage" not in session:
                session["token_usage"] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_cost_usd": 0.0
                }

            session["token_usage"]["input_tokens"] += input_token_count
            session["token_usage"]["output_tokens"] += output_token_count
            input_cost = input_token_count * 0.005 / 1000
            output_cost = output_token_count * 0.015 / 1000
            session["token_usage"]["estimated_cost_usd"] += input_cost + output_cost

            print(f"💵 Input Tokens: {input_token_count} | Output Tokens: {output_token_count}")
            print(f"💰 Session Estimated Cost: ${session['token_usage']['estimated_cost_usd']:.4f}")
        else:
            print(f"💵 Input Tokens: {input_token_count} | Output Tokens: {output_token_count}")
            print("⚠️ No request context, token usage not tracked in session.")

    return generate()

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    user_query = data.get("query", "").strip().lower()
    if not user_query:
        return jsonify({"error": "Query cannot be empty."}), 400

    print(f"🔍 User Query: {user_query}")

    # Reset chat handling
    if "__reset_chat__" in user_query:
        print("🔄 Reset command received. Clearing session.")
        session.clear()
        return jsonify({"message": "Chat session has been reset."}), 200

    if "chat_history" not in session:
        session["chat_history"] = []
    session["chat_history"].append({"role": "user", "content": user_query})
    session["last_user_query"] = user_query

    grouped_results = search_and_filter(user_query)

    ordered_chunks = []
    if not grouped_results and "context_chunks_json" in session:
        print("🔁 Using previous context due to follow-up query.")
        ordered_chunks = sorted(session["context_chunks_json"], key=lambda x: x["chunk_index"])
    else:
        seen_row_ids = set()
        for recipe in grouped_results:
            for r in recipe["chunks"]:
                row_id = int(r["row_id"])
                if row_id not in seen_row_ids:
                    ordered_chunks.append({
                        "row_id": r["row_id"],
                        "chunk_index": r["chunk_index"],
                        "distance_score": r["distance_score"],
                        "text": r.get("text", "[No content found]")
                    })
                    seen_row_ids.add(row_id)
        ordered_chunks.sort(key=lambda x: x["distance_score"])

    # Store result in session
    if "context_chunks_json" not in session:
        session["context_chunks_json"] = []
    existing_ids = {c["row_id"] for c in session["context_chunks_json"]}

    new_context_chunks = []
    for chunk in ordered_chunks:
        if chunk["row_id"] not in existing_ids:
            new_context_chunks.append({
                "row_id": chunk["row_id"],
                "chunk_index": chunk["chunk_index"],
                "distance_score": round(chunk["distance_score"], 4),
                "text": chunk.get("text", "[No content found]")
            })

    session["context_chunks_json"].extend(new_context_chunks)
    print(f"🧠 Context size now: {len(session['context_chunks_json'])}")

    # Token-aware context trim
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_budget = 21000
    context_chunks = []
    total_tokens = 0

    for r in ordered_chunks:
        text = r["text"].strip()
        if "cid:" in text and len(text) < 200:
            continue
        formatted = (
            f"[Row ID: {r['row_id']} | Chunk: {r['chunk_index']} | Score: {r['distance_score']}]\n{text}"
        )
        token_len = len(encoding.encode(formatted))
        if total_tokens + token_len > token_budget:
            break
        context_chunks.append(formatted)
        total_tokens += token_len

    context_text = "\n\n".join(context_chunks)
    print(f"📦 Prepared {len(context_chunks)} context chunks — {total_tokens} tokens total")

    session["chat_history"].append({"role": "assistant", "content": "Generating response..."})
    try:
        return Response(stream_gpt_response(user_query, context_text, session["chat_history"]), content_type='text/event-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/session-cost")
def session_cost():
    usage = session.get("token_usage", {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0
    })
    return jsonify(usage)


# New route to list recipe titles from the database
@app.route("/list-titles")
def list_titles():
    db_path = "recipe_text_chunks.db"  # Update path if your DB is elsewhere
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT(filename) FROM recipe_embeddings")
            rows = cursor.fetchall()
            titles = [row[0] for row in rows]
        return jsonify({"titles": titles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)