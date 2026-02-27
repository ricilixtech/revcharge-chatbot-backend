import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from fastapi.middleware.cors import CORSMiddleware
import time

# =============================
# LOAD ENV
# =============================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)

# =============================
# CREATE FASTAPI APP
# =============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# LOAD RULES
# =============================
def load_rules():
    if os.path.exists("rules.txt"):
        with open("rules.txt", "r", encoding="utf-8") as f:
            return f.read()
    return ""

rules_text = load_rules()

# =============================
# EMBEDDING CACHE FILES
# =============================
FAQ_EMBED_FILE = "faq_embeddings.json"
INVENTORY_EMBED_FILE = "inventory_embeddings.json"

faq_chunks, faq_embeddings = [], []
inventory_chunks, inventory_embeddings = [], []

def embed_with_retry(text, model="gemini-embedding-001", retries=3, delay=5):
    """Generate embedding with retries in case of quota or network errors."""
    for attempt in range(retries):
        try:
            embedding = client.models.embed_content(model=model, contents=text)
            return np.array(embedding.embeddings[0].values)
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    raise Exception("Failed to generate embedding after retries")

def load_faq():
    global faq_chunks, faq_embeddings
    if not os.path.exists("FAQ.txt"):
        return

    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()

    faq_chunks[:] = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    # Load cached embeddings if available
    if os.path.exists(FAQ_EMBED_FILE):
        with open(FAQ_EMBED_FILE, "r") as f:
            faq_embeddings[:] = [np.array(e) for e in json.load(f)]
        return

    # Generate embeddings and cache
    faq_embeddings.clear()
    for chunk in faq_chunks:
        faq_embeddings.append(embed_with_retry(chunk))

    with open(FAQ_EMBED_FILE, "w") as f:
        json.dump([e.tolist() for e in faq_embeddings], f)

def load_inventory():
    global inventory_chunks, inventory_embeddings
    inventory_file = "inventory db chatbot - Sheet1.csv"
    if not os.path.exists(inventory_file):
        return

    df = pd.read_csv(inventory_file)
    inventory_chunks[:] = [
        " | ".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]

    # Load cached embeddings if available
    if os.path.exists(INVENTORY_EMBED_FILE):
        with open(INVENTORY_EMBED_FILE, "r") as f:
            inventory_embeddings[:] = [np.array(e) for e in json.load(f)]
        return

    inventory_embeddings.clear()
    for chunk in inventory_chunks:
        inventory_embeddings.append(embed_with_retry(chunk))

    with open(INVENTORY_EMBED_FILE, "w") as f:
        json.dump([e.tolist() for e in inventory_embeddings], f)

load_faq()
load_inventory()

# =============================
# COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=2):
    if not embeddings:
        return ""
    q_vector = embed_with_retry(question)
    similarities = [(cosine_similarity(q_vector, e), c) for e, c in zip(embeddings, chunks)]
    similarities.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join([item[1] for item in similarities[:top_k]])

def retrieve_relevant_faq(question):
    return retrieve_relevant_chunks(question, faq_chunks, faq_embeddings)

def retrieve_relevant_inventory(question):
    return retrieve_relevant_chunks(question, inventory_chunks, inventory_embeddings)

# =============================
# SESSION-BASED MEMORY
# =============================
chat_sessions = {}

def update_memory(session_id, user_msg, bot_reply):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    chat_sessions[session_id].append({"customer": user_msg, "bot": bot_reply})
    if len(chat_sessions[session_id]) > 5:
        chat_sessions[session_id].pop(0)

def get_memory_text(session_id):
    if session_id not in chat_sessions:
        return ""
    return "\n".join(
        f"Customer: {item['customer']}\nBot: {item['bot']}"
        for item in chat_sessions[session_id]
    )

# =============================
# REQUEST MODEL
# =============================
class ChatRequest(BaseModel):
    message: str
    session_id: str

# =============================
# CHAT ENDPOINT
# =============================
@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    relevant_faq = retrieve_relevant_faq(request.message)
    relevant_inventory = retrieve_relevant_inventory(request.message)
    memory_context = get_memory_text(session_id)

    prompt = f"""
You are a professional customer support assistant.

Follow these rules:
{rules_text}

Previous Conversation:
{memory_context}

Relevant FAQ:
{relevant_faq}

Relevant Inventory Information:
{relevant_inventory}

If answer is not found in the FAQ or Inventory, say:
"I will forward this to a human agent."

Customer Question:
{request.message}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        bot_reply = response.text or "I will forward this to a human agent."
    except Exception as e:
        print("Generation error:", e)
        bot_reply = "Service temporarily unavailable. Please try again."

    update_memory(session_id, request.message, bot_reply)
    return {"reply": bot_reply}