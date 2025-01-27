import os
import uuid
import logging
from logging.handlers import TimedRotatingFileHandler
import time
from typing import Dict

import openai
import pinecone
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from fastapi.responses import JSONResponse

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

# Initialize OpenAI and Pinecone API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., 'us-west1-gcp'

if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.critical("API keys and Pinecone environment must be set as environment variables.")
    raise ValueError("API keys and Pinecone environment must be set as environment variables.")

openai.api_key = OPENAI_API_KEY

try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    logger.info("Pinecone initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Pinecone: {e}", exc_info=True)
    raise

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

# Pydantic Models
class TextInput(BaseModel):
    text: str
    chunk_size: int = 500
    overlap: int = 50

class VectorStoreResponse(BaseModel):
    index_name: str
    dimension: int
    metadata: Dict[str, str]

class DeleteVectorStoreRequest(BaseModel):
    index_name: str

class DeleteVectorStoreResponse(BaseModel):
    message: str
    index_name: str
    
# =======================
# Pydantic Models for Chat
# =======================

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    input: str
    response: str

class SessionStartResponse(BaseModel):
    session_id: str
    
# =======================
# Pydantic Models for Delete Chat
# =======================
class DeleteRequest(BaseModel):
    session_id: str

class DeleteResponse(BaseModel):
    detail: str
    session_id: str
    remaining_sessions: int

# Custom Exception Classes
class PineconeIndexNotFoundError(HTTPException):
    def __init__(self, index_name: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pinecone index '{index_name}' not found."
        )

class PineconeServiceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail
        )

class OpenAIServiceError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail
        )

# Helper Functions
def retry_operation(operation, retries=3, delay=2):
    """
    Retry an operation multiple times with delay.
    """
    for attempt in range(retries):
        try:
            return operation()
        except (pinecone.PineconeException, openai.error.OpenAIError) as e:
            logger.warning(f"Attempt {attempt + 1} failed with error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    logger.error(f"All {retries} attempts failed.")
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="External service is unavailable. Please try again later."
    )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Split text into chunks with specified size and overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    logger.debug(f"Text split into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(text_chunks):
    """
    Generate embeddings for each text chunk using OpenAI.
    """
    embeddings = []
    for i, chunk in enumerate(text_chunks):
        try:
            response = openai.Embedding.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            embeddings.append(embedding)
            logger.debug(f"Generated embedding for chunk {i}.")
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error for chunk {i}: {e}", exc_info=True)
            raise OpenAIServiceError("Failed to generate embeddings.")
        except Exception as e:
            logger.error(f"Unexpected error during embedding generation for chunk {i}: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred during embedding generation.")
    logger.info("All embeddings generated successfully.")
    return embeddings

# Exception Handler for Validation Errors
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# API Endpoints
@app.post("/create-vector-store", response_model=VectorStoreResponse)
def create_vector_store(input: TextInput):
    logger.info("Received request to create a new vector store.")
    try:
        # Step 1: Chunk the text
        text_chunks = chunk_text(input.text, input.chunk_size, input.overlap)
        logger.info(f"Text split into {len(text_chunks)} chunks.")

        # Step 2: Generate embeddings with retry
        def generate():
            return generate_embeddings(text_chunks)
        embeddings = retry_operation(generate)

        # Step 3: Create a unique index name
        unique_id = str(uuid.uuid4())
        index_name = f"text-vector-store-{unique_id}"
        logger.debug(f"Generated unique index name: {index_name}")

        # Step 4: Create Pinecone index with retry
        def create_pinecone_index():
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(index_name, dimension=1536)  # Ada embeddings have 1536 dimensions
                logger.info(f"Pinecone index '{index_name}' created.")
            else:
                logger.warning(f"Pinecone index '{index_name}' already exists.")
        retry_operation(create_pinecone_index)

        index = pinecone.Index(index_name)

        # Step 5: Prepare data for Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            vectors.append((
                f"chunk_{i}",
                embedding,
                {"text": chunk}
            ))
        logger.debug(f"Prepared {len(vectors)} vectors for upsert.")

        # Step 6: Upsert vectors into Pinecone with retry
        def upsert_vectors():
            index.upsert(vectors=vectors)
            logger.info(f"Upserted {len(vectors)} vectors into '{index_name}'.")
        retry_operation(upsert_vectors)

        # Step 7: Prepare metadata to return
        metadata = {
            "index_name": index_name,
            "environment": PINECONE_ENVIRONMENT,
            "dimension": 1536,
            "vector_count": len(vectors)
        }

        return VectorStoreResponse(
            index_name=index_name,
            dimension=1536,
            metadata=metadata
        )

    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he

    except Exception as e:
        logger.error(f"Unexpected error in create_vector_store: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while creating the vector store.")

@app.post("/delete-vector-store", response_model=DeleteVectorStoreResponse)
def delete_vector_store(request: DeleteVectorStoreRequest):
    index_name = request.index_name
    logger.info(f"Received request to delete Pinecone index: {index_name}")
    
    try:
        def check_index_exists():
            if index_name not in pinecone.list_indexes():
                logger.warning(f"Pinecone index '{index_name}' not found.")
                raise PineconeIndexNotFoundError(index_name)
        retry_operation(check_index_exists)

        # Step 1: Delete the index with retry
        def delete_pinecone_index():
            pinecone.delete_index(index_name)
            logger.info(f"Pinecone index '{index_name}' successfully deleted.")
        retry_operation(delete_pinecone_index)

        return DeleteVectorStoreResponse(
            message=f"Pinecone index '{index_name}' has been deleted successfully.",
            index_name=index_name
        )
    
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise he
    
    except Exception as e:
        logger.error(f"Unexpected error while deleting index '{index_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while deleting the index.")

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

        return {
            "session_id": session_id,
            "input": user_message,
            "response": ai_response
        }

    except HTTPException as http_exc:
        # Specific HTTP exceptions are logged at warning level
        logger.warning(f"HTTPException for session {session_id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # General exceptions are logged with stack trace
        logger.exception(f"Error processing chat for session {session_id}")
        raise HTTPException(status_code=500, detail="Failed to process chat message")
    
# =======================
# Delete Endpoint
# =======================
@app.delete("/delete-session", response_model=DeleteResponse)
async def delete_session(request: DeleteRequest):
    """
    Deletes a chat session and its associated memory
    """
    session_id = request.session_id
    logger.info(f"Delete request initiated for session: {session_id}")
    
    try:
        # Validate session existence
        if session_id not in sessions:
            logger.warning(f"Delete attempt for non-existent session: {session_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
                headers={"X-Session-Status": "not_found"}
            )

        # Perform deletion
        del sessions[session_id]
        remaining = len(sessions)
        
        logger.info(f"Successfully deleted session: {session_id}")
        logger.debug(f"Remaining active sessions: {remaining}")
        
        return {
            "detail": "Session deleted successfully",
            "session_id": session_id,
            "remaining_sessions": remaining
        }

    except HTTPException as http_exc:
        logger.error(f"Deletion failed for session {session_id}: {http_exc.detail}")
        raise http_exc
        
    except KeyError as ke:
        error_msg = f"Session key error: {str(ke)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during deletion",
            headers={"X-Error-Type": "key_error"}
        )
        
    except Exception as e:
        error_msg = f"Unexpected error deleting session {session_id}: {str(e)}"
        logger.exception(error_msg)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
            headers={"X-Error-Type": "unexpected_error"}
        )

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