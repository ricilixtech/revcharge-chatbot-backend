import os
import json
import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# =============================
# LOAD ENV
# =============================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# =============================
# LOCAL EMBEDDING MODEL
# =============================
model = SentenceTransformer("all-MiniLM-L6-v2")

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
# FILES
# =============================
FAQ_EMBED_FILE = "faq_embeddings.json"
INVENTORY_EMBED_FILE = "inventory_embeddings.json"

faq_chunks, faq_embeddings = [], []
inventory_chunks, inventory_embeddings = [], []

# =============================
# EMBEDDING FUNCTION
# =============================
def embed(text):
    return model.encode(text)

# =============================
# LOAD FAQ
# =============================
def load_faq():
    global faq_chunks, faq_embeddings

    if not os.path.exists("FAQ.txt"):
        return

    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()

    faq_chunks[:] = [c.strip() for c in content.split("\n\n") if c.strip()]

    if os.path.exists(FAQ_EMBED_FILE):
        with open(FAQ_EMBED_FILE) as f:
            faq_embeddings[:] = [np.array(e) for e in json.load(f)]
        return

    faq_embeddings.clear()

    for chunk in faq_chunks:
        faq_embeddings.append(embed(chunk))

    with open(FAQ_EMBED_FILE, "w") as f:
        json.dump([e.tolist() for e in faq_embeddings], f)

# =============================
# LOAD INVENTORY
# =============================
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

    if os.path.exists(INVENTORY_EMBED_FILE):
        with open(INVENTORY_EMBED_FILE) as f:
            inventory_embeddings[:] = [np.array(e) for e in json.load(f)]
        return

    inventory_embeddings.clear()

    for chunk in inventory_chunks:
        inventory_embeddings.append(embed(chunk))

    with open(INVENTORY_EMBED_FILE, "w") as f:
        json.dump([e.tolist() for e in inventory_embeddings], f)

load_faq()
load_inventory()

# =============================
# COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_chunks(question, chunks, embeddings, top_k=2):

    if not embeddings:
        return ""

    q_vector = embed(question)

    sims = [(cosine_similarity(q_vector, e), c)
            for e, c in zip(embeddings, chunks)]

    sims.sort(reverse=True, key=lambda x: x[0])

    return "\n\n".join([x[1] for x in sims[:top_k]])

def retrieve_relevant_faq(question):
    return retrieve_chunks(question, faq_chunks, faq_embeddings)

def retrieve_relevant_inventory(question):
    return retrieve_chunks(question, inventory_chunks, inventory_embeddings)

# =============================
# SESSION MEMORY
# =============================
chat_sessions = {}

def update_memory(session_id, user_msg, bot_reply):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "customer": user_msg,
        "bot": bot_reply
    })

    if len(chat_sessions[session_id]) > 5:
        chat_sessions[session_id].pop(0)

def get_memory(session_id):

    if session_id not in chat_sessions:
        return ""

    return "\n".join(
        f"Customer: {m['customer']}\nBot: {m['bot']}"
        for m in chat_sessions[session_id]
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
def chat(req: ChatRequest):

    faq = retrieve_relevant_faq(req.message)
    inventory = retrieve_relevant_inventory(req.message)
    memory = get_memory(req.session_id)

    prompt = f"""
You are a professional customer support assistant.

Follow these rules:
{rules_text}

Previous Conversation:
{memory}

Relevant FAQ:
{faq}

Relevant Inventory:
{inventory}

If answer is not found say:
"I will forward this to a human agent."

Customer Question:
{req.message}
"""

    try:

        payload = {
            "model": "arcee-ai/trinity-large-preview:free",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        res = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload
        )

        data = res.json()

        bot_reply = data["choices"][0]["message"]["content"]

    except Exception as e:
        print(e)
        bot_reply = "Service temporarily unavailable."

    update_memory(req.session_id, req.message, bot_reply)

    return {"reply": bot_reply}