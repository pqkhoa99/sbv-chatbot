from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime
import json
from models.database import User, Conversation, Message as DBMessage
from ai_models.sbv_lawgraph import app as lawgraph_app
from ai_models.sbv_bm25 import create_bm25_chain
from ai_models.sbv_naiverag import create_naive_rag_chain
from ai_models.sbv_advancedrag import create_advanced_rag_chain
from ai_models.sbv_gpt import generate_gpt_answer
from ai_models.sbv_gemini import generate_gemini_answer

# Model chains cache
_model_chains = {}

def get_model_chain(model_name: str):
    """Lazy initialization of model chains"""
    if model_name not in _model_chains:
        if model_name == "sbv-bm25":
            _model_chains[model_name] = create_bm25_chain()
        elif model_name == "sbv-naiverag":
            _model_chains[model_name] = create_naive_rag_chain()
        elif model_name == "sbv-advancedrag":
            _model_chains[model_name] = create_advanced_rag_chain()
    return _model_chains.get(model_name)

def get_or_create_user_by_username(db: Session, username: str) -> User:
    """Get or create user by username"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def send_message(db: Session, conversation_id: int, question: str, username: str) -> dict:
    """Process chat message and generate response"""
    user = get_or_create_user_by_username(db, username)
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Save user message
    user_message = DBMessage(
        conversation_id=conversation_id,
        role="user",
        content=question
    )
    db.add(user_message)
    db.commit()
    
    # Generate AI response based on model
    model_name = conversation.model
    documents = None
    
    try:
        if model_name == "sbv-lawgraph":
            result = lawgraph_app.invoke({"question": question})
            answer = result.get("answer", "Không tìm thấy câu trả lời phù hợp.")
            documents = result.get("source_documents", [])
        
        elif model_name in ["sbv-bm25", "sbv-naiverag", "sbv-advancedrag"]:
            chain = get_model_chain(model_name)
            if chain:
                # All these RAG models expect dict input with "question" key
                result = chain.invoke({"question": question})
                    
                if isinstance(result, dict):
                    answer = result.get("answer", result.get("result", str(result)))
                    documents = result.get("source_documents", result.get("documents", result.get("context", [])))
                else:
                    answer = str(result)
            else:
                answer = "Model not available"
        
        elif model_name == "sbv-gpt":
            answer = generate_gpt_answer(question)
        
        elif model_name == "sbv-gemini":
            answer = generate_gemini_answer(question)
        
        else:
            answer = f"Unknown model: {model_name}"
    
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
        documents = None
    
    # Format documents for storage
    docs_to_store = None
    if documents:
        docs_to_store = str([{
            "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
        } for doc in documents[:3]])
    
    # Save assistant message
    assistant_message = DBMessage(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        documents=docs_to_store
    )
    db.add(assistant_message)
    
    # Update conversation timestamp
    conversation.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(assistant_message)
    
    return {
        "message": {
            "id": assistant_message.id,
            "role": "assistant",
            "content": answer,
            "documents": eval(docs_to_store) if docs_to_store else [],
            "created_at": assistant_message.created_at.isoformat()
        },
        "conversation_updated_at": conversation.updated_at.isoformat()
    }

def get_available_models() -> list:
    """Get list of available AI models"""
    return [
        {
            "id": "sbv-lawgraph",
            "name": "SBV LawGraph",
            "description": "Kết hợp các kỹ thuật RAG nâng cao với Knowledge Graph, phân tích chuyên sâu mối quan hệ pháp lý",
            "badge": "Khuyến nghị"
        },
        {
            "id": "sbv-advancedrag",
            "name": "Advanced RAG",
            "description": "RAG nâng cao với Hybrid Search, độ chính xác cao",
            "badge": "Nâng cao"
        },
        {
            "id": "sbv-naiverag",
            "name": "Naive RAG",
            "description": "RAG cơ bản với vector search đơn giản",
            "badge": "Cơ bản"
        },
        {
            "id": "sbv-bm25",
            "name": "BM25 Search",
            "description": "Tìm kiếm từ khóa truyền thống BM25",
            "badge": "Truyền thống"
        },
        {
            "id": "sbv-gpt",
            "name": "GPT 5",
            "description": "OpenAI GPT 5, tri thức tổng quát",
            "badge": "OpenAI Model"
        },
        {
            "id": "sbv-gemini",
            "name": "Gemini 2.5 Pro",
            "description": "Google Gemini 2.5 Pro, đa ngôn ngữ",
            "badge": "Gemini Model"
        }
    ]
