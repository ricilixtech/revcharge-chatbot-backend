import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from fastapi.middleware.cors import CORSMiddleware

# =============================
# LOAD ENV (Works locally + Railway)
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
# LOAD FAQ + INVENTORY + EMBEDDINGS
# =============================
faq_chunks = []
faq_embeddings = []

inventory_chunks = []
inventory_embeddings = []

def load_faq():
    global faq_chunks, faq_embeddings

    if not os.path.exists("FAQ.txt"):
        return

    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()

    faq_chunks[:] = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    faq_embeddings.clear()

    for chunk in faq_chunks:
        embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        faq_embeddings.append(np.array(embedding.embeddings[0].values))


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

    inventory_embeddings.clear()

    for chunk in inventory_chunks:
        embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        inventory_embeddings.append(np.array(embedding.embeddings[0].values))


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

    question_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question
    )

    q_vector = np.array(question_embedding.embeddings[0].values)

    similarities = [(cosine_similarity(q_vector, e), c)
                    for e, c in zip(embeddings, chunks)]

    similarities.sort(reverse=True, key=lambda x: x[0])

    return "\n\n".join([item[1] for item in similarities[:top_k]])

def retrieve_relevant_faq(question):
    return retrieve_relevant_chunks(question, faq_chunks, faq_embeddings)

def retrieve_relevant_inventory(question):
    return retrieve_relevant_chunks(question, inventory_chunks, inventory_embeddings)

# =============================
# SESSION-BASED MEMORY
# =============================
chat_sessions = {}   # session_id -> list of messages

def update_memory(session_id, user_msg, bot_reply):

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_sessions[session_id].append({
        "customer": user_msg,
        "bot": bot_reply
    })

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
    session_id: str   # <-- IMPORTANT

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