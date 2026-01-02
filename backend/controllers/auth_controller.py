from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging
from models.database import get_db
from services.auth_service import register_user, authenticate_user, verify_token

logger = logging.getLogger("auth_controller")
router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str
    name: str

@router.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    logger.info(f"Receive POST /api/auth/register request: {{'username': '{request.username}', 'name': '{request.name}'}}")
    response = register_user(request.username, request.password, request.name, db)
    logger.info(f"Send POST /api/auth/register response: {{'success': True, 'username': '{response['username']}'}}")
    return response

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    logger.info(f"Receive POST /api/auth/login request: {{'username': '{request.username}'}}")
    response = authenticate_user(request.username, request.password, db)
    logger.info(f"Send POST /api/auth/login response: {{'success': True, 'username': '{response['username']}', 'role': '{response['role']}'}}")
    return response

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)
    return payload
