import os
import uuid
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cerebras.cloud.sdk import Cerebras
from langchain.memory import ConversationBufferMemory

import Prompts

# Configure logging
LOG_DIR = os.path.abspath("logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "RDF.log")

# Create handlers
file_handler = TimedRotatingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)

console_handler = logging.StreamHandler()

# Create formatters with more context
detailed_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
)

file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(detailed_formatter)

# Configure root logger
logger = logging.getLogger("RDF")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(detailed_formatter)
logger.addHandler(console_handler)

app = FastAPI()

# =======================
# Initialize Cerebras Client
# =======================

# Fetch API key from environment variables
api_key = os.environ.get("CEREBRAS_API_KEY")
if not api_key:
    logger.error("CEREBRAS_API_KEY environment variable not set")
    raise EnvironmentError("CEREBRAS_API_KEY environment variable not set")

try:
    client = Cerebras(api_key=api_key)
    logger.info("Cerebras client initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Cerebras client")
    raise e

# =======================
# In-Memory Session Storage
# =======================

# Note: For production, replace with a persistent database
sessions: Dict[str, ConversationBufferMemory] = {}

# =======================
# Pydantic Models
# =======================

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

class SessionStartResponse(BaseModel):
    session_id: str

# =======================
# Exception Handling
# =======================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles all uncaught exceptions, logs them, and returns a standardized error response.
    """
    logger.exception(f"Unhandled exception for request {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# =======================
# API Endpoints
# =======================

@app.post("/start-session", response_model=SessionStartResponse)
async def start_session():
    """
    Initializes a new chat session with a unique ID.
    """
    try:
        session_id = str(uuid.uuid4())
        memory = ConversationBufferMemory()
        memory.chat_memory.add_ai_message(Prompts.SYSTEM_PROMPT)
        sessions[session_id] = memory
        logger.info(f"Started new session with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        logger.exception("Error starting new session")
        raise HTTPException(status_code=500, detail="Failed to start session")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Processes a chat message and returns an AI response.
    """
    session_id = request.session_id
    user_message = request.message

    logger.debug(f"Received message for session {session_id}: {user_message}")

    # Validate session ID
    if session_id not in sessions:
        logger.warning(f"Session ID not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    memory = sessions[session_id]

    try:
        # Add user message to memory
        memory.chat_memory.add_user_message(user_message)
        logger.debug(f"Added user message to session {session_id} memory")

        # Get conversation history
        history = memory.load_memory_variables({}).get("history", "")
        logger.debug(f"Loaded history for session {session_id}: {history}")

        # Create prompt with history
        prompt = f"{history}\nUser: {user_message}\nAI:"
        logger.debug(f"Generated prompt for Cerebras: {prompt}")

        # Get AI response from Cerebras
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": Prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model="llama3.1-8b",
        )

        ai_response = response.choices[0].message.content
        logger.debug(f"Received AI response for session {session_id}: {ai_response}")

        # Add AI response to memory
        memory.chat_memory.add_ai_message(ai_response)
        logger.debug(f"Added AI message to session {session_id} memory")

        return {"response": ai_response}

    except HTTPException as http_exc:
        # Specific HTTP exceptions are logged at warning level
        logger.warning(f"HTTPException for session {session_id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # General exceptions are logged with stack trace
        logger.exception(f"Error processing chat for session {session_id}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")

# =======================
# Application Startup
# =======================

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("FastAPI application startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("FastAPI application shutdown initiated.")