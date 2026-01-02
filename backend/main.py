from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from models.database import init_db
from controllers import auth_controller, conversation_controller, chat_controller

logger = logging.getLogger("main")

# FastAPI App
app = FastAPI(title="SBV Legal Chatbot API", version="2.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_controller.router)
app.include_router(conversation_controller.router)
app.include_router(chat_controller.router)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.warning("Application will continue but conversation storage will not work")

@app.get("/")
async def root():
    return {
        "message": "SBV Legal Chatbot API v2.0",
        "version": "2.0.0",
        "features": ["conversation_management", "model_per_conversation", "message_history"],
        "endpoints": {
            "login": "/api/auth/login",
            "register": "/api/auth/register",
            "conversations": "/api/conversations",
            "chat": "/api/chat",
            "models": "/api/models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
