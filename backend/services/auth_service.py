from fastapi import HTTPException, status
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta
from models.database import User

# Configuration
SECRET_KEY = "sbv-chatbot-secret-key-2025"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

# Mock user database (in-memory)
USERS_DB = {
    "admin": {"username": "admin", "password": "admin", "role": "admin", "name": "Administrator"},
    "user": {"username": "user", "password": "user", "role": "user", "name": "Demo User"}
}

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_or_create_user(db: Session, username: str, name: str = None) -> User:
    """Get existing user or create new one"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user = User(username=username, name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
    elif name and not user.name:
        user.name = name
        db.commit()
        db.refresh(user)
    return user

def register_user(username: str, password: str, name: str, db: Session) -> dict:
    """Register a new user"""
    if username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên đăng nhập đã tồn tại"
        )
    
    if len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên đăng nhập phải có ít nhất 3 ký tự"
        )
    
    if len(password) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Mật khẩu phải có ít nhất 3 ký tự"
        )
    
    if not name or len(name.strip()) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Họ và tên phải có ít nhất 2 ký tự"
        )
    
    USERS_DB[username] = {
        "username": username,
        "password": password,
        "role": "user",
        "name": name.strip()
    }
    
    get_or_create_user(db, username, name.strip())
    
    return {"message": "Đăng ký thành công", "username": username}

def authenticate_user(username: str, password: str, db: Session) -> dict:
    """Authenticate user and return token"""
    user = USERS_DB.get(username)
    
    if not user or user["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    get_or_create_user(db, user["username"], user.get("name"))
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
        "name": user.get("name", user["username"])
    }
