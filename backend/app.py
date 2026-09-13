from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str
    language: str
    history: Optional[List[dict]] = []

def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI is not configured. Add GEMINI_API_KEY to .env, then restart the server."
        )
    genai.configure(api_key=api_key)
    return genai

def build_prompt(message: str, language: str, history: list) -> str:
    lang_name = {
        "en": "English",
        "hi": "Hindi",
        "kn": "Kannada",
        "ta": "Tamil",
        "te": "Telugu",
        "mr": "Marathi",
        "bn": "Bengali",
        "gu": "Gujarati",
        "pa": "Punjabi",
    }.get(language, "English")

    time_str = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p %Z")

    prompt = f"""You are BIS Saathi, an AI assistant for the Bureau of Indian Standards (BIS), India.

Current time: {time_str}

Language: {lang_name}

About BIS (Bureau of Indian Standards):
- BIS is the National Standards Body of India, under the Ministry of Consumer Affairs, Food & Public Distribution.
- It formulates and publishes Indian Standards (IS) for products, processes, and services.
- BIS runs product certification schemes (including the ISI Mark) to assure quality and safety for consumers.
- BIS also supports consumer protection, testing, and standards-related information services.
- Official website: https://bis.gov.in

Your job:
- Answer questions about BIS services, Indian Standards, certification (like ISI Mark), registrations, and consumer support.
- Be clear, factual, and student/consumer-friendly.
- Use the same language as the user's query ({lang_name}).
- If something depends on latest rules/fees/forms, explain the general process and advise checking the official BIS website or local BIS office for the most current details.
- Do NOT invent facts about BIS or Indian Standards.

Conversation history (most recent last):
{json.dumps(history, indent=2, ensure_ascii=False) if history else "No prior messages."}

User's latest message:
{message}

Respond in {lang_name}, clearly and helpfully.
"""
    return prompt

@app.get("/")
def root():
    return {"message": "BIS Saathi API is running. Visit /api/health for status."}

@app.get("/api/health")
def health():
    try:
        client = get_genai_client()
        return {"status": "ok", "ai": "configured"}
    except Exception as e:
        return {"status": "ok", "ai": "error", "detail": str(e)}

@app.get("/api/models")
def list_models():
    try:
        client = get_genai_client()
        models = list(client.list_models())
        result = []
        for m in models:
            result.append({
                "name": getattr(m, "name", None),
                "display_name": getattr(m, "display_name", None),
                "supported_methods": getattr(m, "supported_generation_methods", None),
            })
        return {"models": result}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to list models: {e}")

@app.post("/api/chat")
def chat(req: ChatMessage):
    try:
        client = get_genai_client()
        model = client.GenerativeModel("gemini-3.6-flash")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI client error: {e}")

    try:
        prompt = build_prompt(req.message, req.language, req.history or [])
        response = model.generate_content(prompt)
        text = response.text.strip() if response.text else "(No response from AI)"
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI service error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)