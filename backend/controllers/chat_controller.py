from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import logging
from models.database import get_db
from services.chat_service import send_message, get_available_models
from controllers.auth_controller import get_current_user

logger = logging.getLogger("chat_controller")
router = APIRouter(prefix="/api", tags=["Chat"])

class ChatRequest(BaseModel):
    conversation_id: int
    question: str

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    documents: Optional[List[dict]] = []
    created_at: str

class ChatResponse(BaseModel):
    message: MessageResponse
    conversation_updated_at: str

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Send a message and get AI response"""
    logger.info(f"Receive POST /api/chat request: {{'conversation_id': {request.conversation_id}, 'question_length': {len(request.question)}, 'user': '{current_user['sub']}'}}")
    response = send_message(db, request.conversation_id, request.question, current_user["sub"])
    logger.info(f"Send POST /api/chat response: {{'message_id': {response['message']['id']}, 'content_length': {len(response['message']['content'])}, 'documents_count': {len(response['message']['documents'])}}}")
    return response

@router.get("/models")
async def get_models(current_user: dict = Depends(get_current_user)):
    """Get list of available AI models"""
    logger.info(f"Receive GET /api/models request: {{'user': '{current_user['sub']}'}}")
    response = get_available_models()
    logger.info(f"Send GET /api/models response: {{'models_count': {len(response)}}}")
    return response
