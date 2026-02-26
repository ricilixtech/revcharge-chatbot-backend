import os
import numpy as np
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
    allow_origins=["*"],  # change to your frontend domain in production
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
# LOAD FAQ + CREATE EMBEDDINGS (SAFE VERSION)
# =============================
faq_chunks = []
faq_embeddings = []

def load_faq():
    global faq_chunks, faq_embeddings

    if not os.path.exists("FAQ.txt"):
        print("FAQ.txt not found")
        return

    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()

    faq_chunks = [
        chunk.strip()
        for chunk in content.split("\n\n")
        if chunk.strip()
    ]

    faq_embeddings.clear()

    for chunk in faq_chunks:
        try:
            embedding = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
            vector = np.array(embedding.embeddings[0].values)
            faq_embeddings.append(vector)
        except Exception as e:
            print("Embedding error:", e)

# Load once at startup
load_faq()

# =============================
# HELPER: COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# RETRIEVE RELEVANT FAQ
# =============================
def retrieve_relevant_faq(question, top_k=2):

    if not faq_embeddings:
        return ""

    try:
        question_embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        q_vector = np.array(question_embedding.embeddings[0].values)

        similarities = []
        for i, faq_vector in enumerate(faq_embeddings):
            score = cosine_similarity(q_vector, faq_vector)
            similarities.append((score, faq_chunks[i]))

        similarities.sort(reverse=True, key=lambda x: x[0])
        return "\n\n".join([item[1] for item in similarities[:top_k]])

    except Exception as e:
        print("Retrieval error:", e)
        return ""

# =============================
# SIMPLE MEMORY (Last 5 Exchanges)
# =============================
chat_memory = []

def update_memory(user_msg, bot_reply):
    chat_memory.append({
        "customer": user_msg,
        "bot": bot_reply
    })

    if len(chat_memory) > 5:
        del chat_memory[0]

def get_memory_text():
    return "\n".join(
        f"Customer: {item['customer']}\nBot: {item['bot']}"
        for item in chat_memory
    )

# =============================
# REQUEST MODEL
# =============================
class ChatRequest(BaseModel):
    message: str

# =============================
# CHAT ENDPOINT
# =============================
@app.post("/chat")
def chat(request: ChatRequest):

    relevant_faq = retrieve_relevant_faq(request.message)
    memory_context = get_memory_text()

    prompt = f"""
You are a professional customer support assistant.

Follow these rules:
{rules_text}

Previous Conversation:
{memory_context}

Relevant FAQ:
{relevant_faq}

If answer is not found in the FAQ, say:
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

    update_memory(request.message, bot_reply)

    return {"reply": bot_reply}