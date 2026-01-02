#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         SBV Legal Chatbot - Stop Application              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    echo -e "${YELLOW}Checking port ${port} for ${name}...${NC}"
    PID=$(lsof -ti:${port})
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}Killing process on port ${port} (PID: ${PID})${NC}"
        kill -9 $PID 2>/dev/null
        sleep 1
        echo -e "${GREEN}✓${NC} ${name} stopped"
    else
        echo -e "${GREEN}✓${NC} No ${name} process running on port ${port}"
    fi
}

# Stop backend and frontend
kill_port 8000 "Backend"
kill_port 3000 "Frontend"

echo ""

# Stop Docker containers
echo -e "${CYAN}Stopping Docker containers...${NC}"
if command -v docker-compose &> /dev/null; then
    docker-compose stop
    echo -e "${GREEN}✓${NC} Docker containers stopped (PostgreSQL, Neo4j, Qdrant)"
    echo -e "${YELLOW}Note: Containers are stopped but not removed. Data is preserved.${NC}"
    echo -e "${YELLOW}To remove containers and volumes: docker-compose down -v${NC}"
else
    echo -e "${YELLOW}Docker Compose not found, skipping...${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Application stopped successfully! ✓               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
