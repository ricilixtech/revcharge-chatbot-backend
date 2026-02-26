import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from fastapi.middleware.cors import CORSMiddleware

# =============================
# LOAD ENV + GEMINI
# =============================
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
# LOAD FAQ + CREATE EMBEDDINGS
# =============================
faq_chunks = []
faq_embeddings = []

def load_faq():
    global faq_chunks, faq_embeddings
    
    if not os.path.exists("FAQ.txt"):
        return

    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()

    faq_chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

    for chunk in faq_chunks:
        embedding = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        faq_embeddings.append(np.array(embedding.embeddings[0].values))

load_faq()

# =============================
# HELPER: COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =============================
# RETRIEVE TOP RELEVANT FAQ
# =============================
def retrieve_relevant_faq(question, top_k=2):
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

# =============================
# MEMORY (Last 5 Exchanges)
# =============================
chat_memory = []  # list of {"customer": "...", "bot": "..."}

def update_memory(user_msg, bot_reply):
    global chat_memory
    
    chat_memory.append({
        "customer": user_msg,
        "bot": bot_reply
    })

    # Keep only last 5 exchanges
    if len(chat_memory) > 5:
        chat_memory = chat_memory[-5:]

def get_memory_text():
    conversation = ""
    for item in chat_memory:
        conversation += f"""
Customer: {item['customer']}
Bot: {item['bot']}
"""
    return conversation.strip()

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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    bot_reply = response.text

    # Update memory
    update_memory(request.message, bot_reply)

    return {"reply": bot_reply}