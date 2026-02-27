import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai.errors import ClientError
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
    allow_origins=["*"],  # Change to frontend domain in production
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
faq_chunks, faq_embeddings = [], []
inventory_chunks, inventory_embeddings = [], []

def embed_text_chunks(chunks, embeddings_list):
    embeddings_list.clear()
    for chunk in chunks:
        try:
            emb = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
            embeddings_list.append(np.array(emb.embeddings[0].values))
        except ClientError as e:
            print("Embedding API error:", e)
        except Exception as e:
            print("Embedding error:", e)

def load_faq():
    global faq_chunks
    if not os.path.exists("FAQ.txt"):
        print("FAQ.txt not found")
        return
    with open("FAQ.txt", "r", encoding="utf-8") as f:
        content = f.read()
    faq_chunks[:] = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    embed_text_chunks(faq_chunks, faq_embeddings)

def load_inventory():
    global inventory_chunks
    inventory_file = "inventory db chatbot - Sheet1.csv"
    if not os.path.exists(inventory_file):
        print("Inventory CSV not found")
        return
    df = pd.read_csv(inventory_file)
    inventory_chunks[:] = [
        " | ".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]
    embed_text_chunks(inventory_chunks, inventory_embeddings)

# Load at startup
load_faq()
load_inventory()

# =============================
# HELPER: COSINE SIMILARITY
# =============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=2):
    if not embeddings:
        return ""
    try:
        q_emb = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )
        q_vector = np.array(q_emb.embeddings[0].values)
        similarities = [(cosine_similarity(q_vector, e), c) for e, c in zip(embeddings, chunks)]
        similarities.sort(reverse=True, key=lambda x: x[0])
        return "\n\n".join([item[1] for item in similarities[:top_k]])
    except ClientError as e:
        print("Embedding error:", e)
        return ""
    except Exception as e:
        print("Embedding error:", e)
        return ""

def retrieve_relevant_faq(question):
    return retrieve_relevant_chunks(question, faq_chunks, faq_embeddings)

def retrieve_relevant_inventory(question):
    return retrieve_relevant_chunks(question, inventory_chunks, inventory_embeddings)

# =============================
# MEMORY (Last 5 Exchanges)
# =============================
chat_memory = []

def update_memory(user_msg, bot_reply):
    chat_memory.append({"customer": user_msg, "bot": bot_reply})
    if len(chat_memory) > 5:
        del chat_memory[0]

def get_memory_text():
    return "\n".join(f"Customer: {item['customer']}\nBot: {item['bot']}" for item in chat_memory)

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
    relevant_inventory = retrieve_relevant_inventory(request.message)
    memory_context = get_memory_text()

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
            contents=prompt
        )
        bot_reply = response.text or "I will forward this to a human agent."
    except ClientError as e:
        print("API ClientError:", e)
        bot_reply = "Service temporarily unavailable: quota exceeded or invalid API key."
    except Exception as e:
        print("Generation error:", e)
        bot_reply = "Service temporarily unavailable. Please try again."

    update_memory(request.message, bot_reply)
    return {"reply": bot_reply}

# =============================
# START WITH RAILWAY PORT
# =============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)