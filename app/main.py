from fastapi import FastAPI

app = FastAPI(title="RAG Chatbot API", version="0.1.0")

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "ok"}

from app.api.v1 import auth_router, documents_router, chat_router

app.include_router(auth_router.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(documents_router.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(chat_router.router, prefix="/api/v1/chat", tags=["Chat"])
