from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime
from models.database import User, Conversation, Message as DBMessage

def get_or_create_user_by_username(db: Session, username: str) -> User:
    """Get or create user by username"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def create_conversation(db: Session, username: str, title: str, model: str) -> dict:
    """Create a new conversation"""
    user = get_or_create_user_by_username(db, username)
    
    conversation = Conversation(
        user_id=user.id,
        title=title,
        model=model
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "model": conversation.model,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "message_count": 0
    }

def get_user_conversations(db: Session, username: str) -> list:
    """Get all conversations for a user"""
    user = get_or_create_user_by_username(db, username)
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    return [{
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
        "message_count": len(conv.messages)
    } for conv in conversations]

def get_conversation_detail(db: Session, conversation_id: int, username: str) -> dict:
    """Get conversation details with messages"""
    user = get_or_create_user_by_username(db, username)
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    messages = [{
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "documents": eval(msg.documents) if msg.documents else [],
        "created_at": msg.created_at.isoformat()
    } for msg in conversation.messages]
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "model": conversation.model,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "messages": messages
    }

def delete_conversation(db: Session, conversation_id: int, username: str) -> dict:
    """Delete a conversation"""
    user = get_or_create_user_by_username(db, username)
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    db.delete(conversation)
    db.commit()
    
    return {"message": "Conversation deleted successfully"}

def update_conversation_title(db: Session, conversation_id: int, username: str, new_title: str) -> dict:
    """Update conversation title"""
    user = get_or_create_user_by_username(db, username)
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation.title = new_title
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conversation)
    
    return {
        "id": conversation.id,
        "title": conversation.title,
        "updated_at": conversation.updated_at.isoformat()
    }
