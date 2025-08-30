#!/usr/bin/env python3
"""
FastAPI service for Immigration Guardian RAG Chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import pathlib

from app.rag_chatbot import ImmigrationRAGChatbot

app = FastAPI(
    title="Immigration Guardian RAG Chatbot",
    description="AI-powered immigration law assistant using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = pathlib.Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize chatbot
chatbot = ImmigrationRAGChatbot()

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = "llama3.2:latest"

class ChatResponse(BaseModel):
    query: str
    visa_type: str
    answer: str
    sources: List[Dict]
    num_sources: int

@app.get("/")
async def root():
    return {
        "message": "Immigration Guardian RAG Chatbot API",
        "status": "running",
        "available_visa_types": ["F1", "F2", "H1B", "H4", "J1", "J2"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the immigration assistant"""
    try:
        response = chatbot.chat(request.query)
        return ChatResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": chatbot.model_name}

@app.get("/visa-types")
async def get_visa_types():
    """Get available visa types"""
    return {"visa_types": chatbot.visa_types}

@app.get("/ui")
async def get_ui():
    """Serve the web interface"""
    static_index = pathlib.Path(__file__).parent / "static" / "index.html"
    if static_index.exists():
        return FileResponse(static_index)
    else:
        raise HTTPException(status_code=404, detail="UI not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
