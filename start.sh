#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   SBV Legal Chatbot - Start Application Script v2.0       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    echo -e "${YELLOW}Checking port ${port} for ${name}...${NC}"
    PID=$(lsof -ti:${port})
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Killing existing process on port ${port} (PID: ${PID})${NC}"
        kill -9 $PID 2>/dev/null
        sleep 1
        echo -e "${GREEN}✓${NC} Port ${port} is now free"
    else
        echo -e "${GREEN}✓${NC} Port ${port} is already free"
    fi
}

# Check if Docker is installed
echo -e "${CYAN}[SETUP 1/6]${NC} Checking Docker installation..."
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker and Docker Compose are installed"
else
    echo -e "${RED}✗${NC} Docker or Docker Compose is not installed"
    echo -e "${YELLOW}Please install Docker Desktop first:${NC}"
    echo "  macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "  Ubuntu: sudo apt install docker.io docker-compose"
    exit 1
fi

# Check if Docker is running
echo -e "${CYAN}[SETUP 2/6]${NC} Checking Docker daemon..."
if ! docker info &> /dev/null; then
    echo -e "${RED}✗${NC} Docker daemon is not running"
    echo -e "${YELLOW}Please start Docker Desktop first${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker daemon is running"

# Start PostgreSQL container
echo -e "${CYAN}[SETUP 3/6]${NC} Starting PostgreSQL container..."
docker-compose up -d postgres
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
sleep 5

# Wait for PostgreSQL to be healthy
for i in {1..30}; do
    if docker exec sbv_postgres pg_isready -U postgres &> /dev/null; then
        echo -e "${GREEN}✓${NC} PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗${NC} PostgreSQL failed to start"
        docker-compose logs postgres
        exit 1
    fi
    sleep 1
done

# Setup Python virtual environment
echo -e "${CYAN}[SETUP 4/6]${NC} Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo -e "${CYAN}[SETUP 5/6]${NC} Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Python dependencies installed"

# Initialize database
echo -e "${CYAN}[SETUP 6/6]${NC} Initializing database tables..."
if [ -f "backend/models/database.py" ]; then
    cd backend
    python -m models.database 2>/dev/null
    cd ..
    echo -e "${GREEN}✓${NC} Database tables initialized"
else
    echo -e "${YELLOW}Note: backend/models/database.py not found, skipping...${NC}"
fi

# Install frontend dependencies (moved before database init)
if [ -d "frontend" ]; then
    cd frontend
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies (this may take a moment)...${NC}"
        npm install > /dev/null 2>&1
        echo -e "${GREEN}✓${NC} Frontend dependencies installed"
    else
        echo -e "${GREEN}✓${NC} Frontend dependencies already installed"
    fi
    cd ..
else
    echo -e "${YELLOW}Note: frontend directory not found, skipping...${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup completed successfully! ✓               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Start other Docker services (Neo4j, Qdrant)
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Starting Docker Services (Neo4j, Qdrant)         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Starting Neo4j and Qdrant containers...${NC}"
docker-compose up -d neo4j qdrant
sleep 3
echo -e "${GREEN}✓${NC} Neo4j running at: ${CYAN}http://localhost:7474${NC}"
echo -e "${GREEN}✓${NC} Qdrant running at: ${CYAN}http://localhost:6333${NC}"
echo ""

# Kill existing processes
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║             Stopping Existing Processes                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
kill_port 8000 "Backend"
kill_port 3000 "Frontend"
echo ""

# Check if .env exists in root directory
if [ ! -f ".env" ]; then
    echo -e "${RED}⚠ Warning: .env not found in root directory!${NC}"
    echo -e "${YELLOW}Please create .env with required variables:${NC}"
    echo "  DATABASE_URL=postgresql://postgres:postgres@localhost:5432/sbv_chatbot"
    echo "  LLM_PROVIDER=openai"
    echo "  OPENAI_API_KEY=your_key_here"
    echo "  GOOGLE_API_KEY=your_key_here"
    echo "  QDRANT_URL=http://localhost:6333"
    echo "  COLLECTION_NAME=sbv_legal_articles"
    echo ""
    echo -e "${YELLOW}See .env.example for all configuration options${NC}"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓${NC} Environment file found: .env"
fi

# Start backend
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                Starting Backend Server                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Starting FastAPI backend on port 8000...${NC}"
cd backend
source ../.venv/bin/activate
# Load environment variables from root .env
export $(grep -v '^#' ../.env | xargs)
nohup python main.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..
sleep 3

# Check if backend started
if ps -p $BACKEND_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} Backend started successfully (PID: ${BACKEND_PID})"
    echo -e "${GREEN}✓${NC} Backend running at: ${CYAN}http://localhost:8000${NC}"
    echo -e "${GREEN}✓${NC} API docs at: ${CYAN}http://localhost:8000/docs${NC}"
else
    echo -e "${RED}✗${NC} Backend failed to start. Check backend/backend.log"
    exit 1
fi

echo ""

# Start frontend
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                Starting Frontend Server                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Starting React frontend on port 3000...${NC}"
cd frontend
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 5

# Check if frontend started
if ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} Frontend started successfully (PID: ${FRONTEND_PID})"
    echo -e "${GREEN}✓${NC} Frontend running at: ${CYAN}http://localhost:3000${NC}"
else
    echo -e "${RED}✗${NC} Frontend failed to start. Check frontend/frontend.log"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Application Started Successfully! 🚀              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}📱Docker Containers:${NC}"
echo "  PostgreSQL: sbv_postgres (port 5432)"
echo "  Neo4j:      sbv_neo4j (ports 7474, 7687)"
echo "  Qdrant:     sbv_qdrant (port 6333)"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  Backend:    backend/backend.log"
echo "  Frontend:   frontend/frontend.log"
echo "  PostgreSQL: docker-compose logs postgres"
echo "  Neo4j:      docker-compose logs neo4j"
echo "  Qdrant:     docker-compose logs qdrant"
echo ""
echo -e "${YELLOW}To stop the application:${NC}"
echo "  ./stop.sh
echo ""
echo -e "${YELLOW}Process IDs:${NC}"
echo "  Backend PID:  ${BACKEND_PID}"
echo "  Frontend PID: ${FRONTEND_PID}"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  Backend:  backend/backend.log"
echo "  Frontend: frontend/frontend.log"
echo ""
echo -e "${YELLOW}To stop the application:${NC}"
echo "  kill ${BACKEND_PID} ${FRONTEND_PID}"
echo "  or run: lsof -ti:8000,3000 | xargs kill -9"
echo ""
echo -e "${GREEN}Happy chatting! 💬${NC}"
echo ""
