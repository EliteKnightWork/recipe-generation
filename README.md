# Recipe Generation

A recipe generation application that creates recipes from ingredients using a T5 model, with optional language enhancement via TinyLlama.

## Prerequisites

### Manual Setup
- Python 3.8+
- Node.js 18+
- npm or yarn

### Docker Setup
- Docker
- Docker Compose
- NVIDIA Container Toolkit (optional, for GPU support)

## Project Structure

```
recipe-generation/
├── backend/          # FastAPI backend
│   ├── main.py
│   ├── requirements.txt
│   └── utils/
├── frontend/         # Next.js frontend
│   └── package.json
├── models/           # Pre-downloaded models
└── .env              # Configuration file
```

## Configuration

Copy `.env.example` to `.env` or edit the existing `.env` file:

```env
# Backend Configuration
BACKEND_PORT=8000

# Frontend Configuration
FRONTEND_PORT=3000

# API URL (for frontend to connect to backend)
API_URL=http://localhost:8000

# GPU Configuration (set to "true" to enable GPU)
USE_GPU=false

# Language Enhancement (set to "true" to enable Llama-based recipe enhancement)
USE_LANGUAGE_ENHANCEMENT=true

# Language Model Selection
LLAMA_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Installation

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Running with Docker

### CPU Mode

```bash
docker-compose up --build
```

### GPU Mode

```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Run in Background

```bash
docker-compose up -d --build
```

### Stop Containers

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f
```

The services will be available at:
- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `POST /generate_recipes` - Generate recipes from ingredients

### Example Request

```bash
curl -X POST "http://localhost:8000/generate_recipes" \
  -H "Content-Type: application/json" \
  -d '["chicken", "garlic", "olive oil", "lemon"]'
```

## Features

- **T5 Recipe Generation**: Uses a fine-tuned T5 model for recipe generation
- **Language Enhancement**: Optional TinyLlama integration for improved recipe descriptions
- **GPU Support**: Configurable GPU acceleration for faster inference
- **Recipe Scoring**: Automatic quality scoring for generated recipes
