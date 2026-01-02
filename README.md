# SBV Legal Chatbot

**Chatbot Hỏi Đáp Pháp Luật Ngân Hàng Nhà Nước Việt Nam**

A chatbot application for the State Bank of Vietnam Legal Documents Question-Answering using RAG and custom Knowledge Graph.

## Features

- 6 AI Models: LawGraph (recommended), Advanced RAG, Naive RAG, BM25, GPT-5, Gemini 2.5 Pro
- Knowledge Graph with Neo4j
- Conversation management
- JWT authentication
- React frontend with syntax highlighting
- Docker containers for dependencies

## Project Structure

```
sbv-chatbot/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── controllers/         # API endpoints
│   ├── services/            # Business logic
│   ├── models/              # Database models
│   └── ai_models/           # AI model implementations
├── frontend/
│   ├── src/
│   │   ├── pages/           # Chat, Login pages
│   │   ├── services/        # API client
│   │   └── contexts/        # Auth context
│   └── package.json
├── data/                     # Legal documents and test data
├── indexing/                 # Database building scripts
├── docker-compose.yml        # Database services
├── start.sh                  # Start script
└── stop.sh                   # Stop script
```

## How to Run

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- OpenAI API Key
- Google API Key

### Quick Start

```bash
# Run the automated start script
./start.sh
```

The script will:
- Start Docker containers (PostgreSQL, Neo4j, Qdrant)
- Setup Python virtual environment
- Install dependencies
- Initialize database
- Start backend server (port 8000)
- Start frontend server (port 3000)

Access the application at `http://localhost:3000`

### Manual Start

1. **Start Docker services**
```bash
docker-compose up -d
```

2. **Configure environment**
Create `.env` following the `.env.example`:

3. **Start backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

4. **Start frontend**
```bash
cd frontend
npm install
npm run dev
```

### Stop Application

```bash
./stop.sh
```

### Default Login

| Username | Password |
|----------|----------|
| admin    | admin    |
| user     | user     |

---

**Version**: 1.0.0