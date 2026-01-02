from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging
from models.database import get_db
from services.conversation_service import (
    create_conversation,
    get_user_conversations,
    get_conversation_detail,
    delete_conversation,
    update_conversation_title
)
from controllers.auth_controller import get_current_user

logger = logging.getLogger("conversation_controller")
router = APIRouter(prefix="/api/conversations", tags=["Conversations"])

class ConversationCreate(BaseModel):
    title: str
    model: str

class ConversationResponse(BaseModel):
    id: int
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int

class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    documents: Optional[List[dict]] = []
    created_at: str

class ConversationDetail(BaseModel):
    id: int
    title: str
    model: str
    created_at: str
    updated_at: str
    messages: List[MessageResponse]

class TitleUpdate(BaseModel):
    title: str

@router.post("", response_model=ConversationResponse)
async def create_new_conversation(
    request: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new conversation"""
    logger.info(f"Receive POST /api/conversations request: {{'title': '{request.title}', 'model': '{request.model}', 'user': '{current_user['sub']}'}}")
    response = create_conversation(db, current_user["sub"], request.title, request.model)
    logger.info(f"Send POST /api/conversations response: {{'id': {response['id']}, 'title': '{response['title']}'}}")
    return response

@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all conversations for the current user"""
    logger.info(f"Receive GET /api/conversations request: {{'user': '{current_user['sub']}'}}")
    response = get_user_conversations(db, current_user["sub"])
    logger.info(f"Send GET /api/conversations response: {{'conversations_count': {len(response)}}}")
    return response

@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get conversation details with messages"""
    logger.info(f"Receive GET /api/conversations/{conversation_id} request: {{'user': '{current_user['sub']}'}}")
    response = get_conversation_detail(db, conversation_id, current_user["sub"])
    logger.info(f"Send GET /api/conversations/{conversation_id} response: {{'id': {response['id']}, 'messages_count': {len(response['messages'])}}}")
    return response

@router.delete("/{conversation_id}")
async def delete_conv(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a conversation"""
    logger.info(f"Receive DELETE /api/conversations/{conversation_id} request: {{'user': '{current_user['sub']}'}}")
    response = delete_conversation(db, conversation_id, current_user["sub"])
    logger.info(f"Send DELETE /api/conversations/{conversation_id} response: {{'success': True}}")
    return response

@router.patch("/{conversation_id}/title")
async def update_title(
    conversation_id: int,
    request: TitleUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update conversation title"""
    logger.info(f"Receive PATCH /api/conversations/{conversation_id}/title request: {{'title': '{request.title}', 'user': '{current_user['sub']}'}}")
    response = update_conversation_title(db, conversation_id, current_user["sub"], request.title)
    logger.info(f"Send PATCH /api/conversations/{conversation_id}/title response: {{'success': True}}")
    return response
