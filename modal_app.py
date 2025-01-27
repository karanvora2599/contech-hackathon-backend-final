import os
import uuid
import logging
import time
from typing import Dict, Optional

import modal
from pydantic import BaseModel

# =======================
# Define the FastAPI Image
# =======================
asgi_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "uvicorn==0.22.0",
        "pydantic==2.9.2",
        "openai",
        "pinecone-client",
        "langchain",
        "cerebras-cloud-sdk",
        "python-dotenv",
        "starlette==0.41.2",
        "pandas",
        "numpy",
        "boto3",
        "awswrangler"
    )
)

# =======================
# Define the Modal App
# =======================
app = modal.App("ExtendedFastAPIEndpoint")

# =======================
# Import Necessary Modules for the FastAPI Application
# =======================
with asgi_image.imports():
    import openai
    import pinecone
    from fastapi import FastAPI, HTTPException, Request, Depends, status
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.exceptions import RequestValidationError
    from pydantic import BaseModel, validator
    from langchain.memory import ConversationBufferMemory
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Pinecone as LangPinecone
    from langchain.chains import ConversationalRetrievalChain
    from cerebras.cloud.sdk import Cerebras
    from logging.handlers import TimedRotatingFileHandler
    from utilities.utility import (
        create_session_with_credentials,
        create_session_with_env_credentials
    )
    from utilities.IMSHandler import IMSQueryHandler
    import Prompts
    from dotenv import load_dotenv

# =======================
# Load Environment Variables
# =======================
load_dotenv()

# =======================
# Configure Logging
# =======================
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
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =======================
# Initialize Cerebras Client
# =======================

# Fetch API key from environment variables
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    logger.error("CEREBRAS_API_KEY environment variable not set")
    raise EnvironmentError("CEREBRAS_API_KEY environment variable not set")

try:
    client = Cerebras(api_key=CEREBRAS_API_KEY)
    logger.info("Cerebras client initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Cerebras client")
    raise e

# =======================
# Initialize Pinecone
# =======================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., 'us-west1-gcp'

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.critical("API keys and Pinecone environment must be set as environment variables.")
    raise ValueError("API keys and Pinecone environment must be set as environment variables.")

try:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    logger.info("Pinecone initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Pinecone: {e}", exc_info=True)
    raise e

# =======================
# In-Memory Session Storage
# =======================

# Note: For production, replace with a persistent database
sessions: Dict[str, Dict] = {}

# =======================
# Cerebras LLM Wrapper
# =======================

class CerebrasLLM:
    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt: str, system_prompt: Optional[str] = None):
        # Use the provided system prompt or fallback to the default
        system_content = system_prompt if system_prompt else Prompts.SYSTEM_PROMPT
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            ai_response = response.choices[0].message.content
            logger.debug(f"Cerebras AI response: {ai_response}")
            return ai_response
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise OpenAIServiceError("Failed to generate AI response.")
        except Exception as e:
            logger.exception(f"Unexpected error in CerebrasLLM: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate AI response.")

class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    def __init__(self, llm, retriever, memory, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(llm=llm, retriever=retriever, memory=memory, **kwargs)
        self.system_prompt = system_prompt

    def _call(self, inputs, run_manager=None):
        # Override the _call method to inject the custom system prompt
        if self.system_prompt:
            self.llm.system_prompt = self.system_prompt
        return super()._call(inputs, run_manager)

# =======================
# Pydantic Models
# =======================

# Text Input for Vector Store Creation
class TextInput(BaseModel):
    text: str
    chunk_size: int = 500
    overlap: int = 50

# Vector Store Response Model
class VectorStoreResponse(BaseModel):
    index_name: str
    dimension: int
    metadata: Dict[str, str]

# Delete Vector Store Request
class DeleteVectorStoreRequest(BaseModel):
    index_name: str

# Delete Vector Store Response
class DeleteVectorStoreResponse(BaseModel):
    message: str
    index_name: str

# Session Models
class SessionStartRequest(BaseModel):
    # Optional: Add any parameters needed for session initialization
    pass

class SessionStartResponse(BaseModel):
    session_id: str

# Chat Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None  # Optional system prompt

    @validator('system_prompt')
    def validate_system_prompt(cls, v):
        if v and len(v) > 1000:
            raise ValueError("System prompt must be 1000 characters or fewer.")
        return v

class ChatResponse(BaseModel):
    session_id: str
    input: str
    response: str

# Delete Session Models
class DeleteRequest(BaseModel):
    session_id: str

class DeleteResponse(BaseModel):
    detail: str
    session_id: str
    remaining_sessions: int

# RAG Chat Models
class ChatRagRequest(BaseModel):
    session_id: str
    vectorstore_id: str
    message: str
    system_prompt: Optional[str] = None  # Optional system prompt

    @validator('vectorstore_id')
    def validate_vectorstore_id(cls, v):
        if not v.startswith("text-vector-store-"):
            raise ValueError("Invalid vectorstore_id format.")
        return v

    @validator('message')
    def validate_message(cls, v):
        if len(v) > 1000:
            raise ValueError("Message length exceeds 1000 characters.")
        return v

    @validator('system_prompt')
    def validate_system_prompt(cls, v):
        if v and len(v) > 1000:
            raise ValueError("System prompt must be 1000 characters or fewer.")
        return v

class ChatRagResponse(BaseModel):
    session_id: str
    vectorstore_id: str
    input: str
    response: str

# =======================
# Custom Exception Classes
# =======================

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

# =======================
# Helper Functions
# =======================

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

def load_vectorstore(vectorstore_id: str):
    """
    Load a Pinecone vector store based on the vectorstore_id.
    """
    try:
        if vectorstore_id not in pinecone.list_indexes():
            logger.error(f"Pinecone index '{vectorstore_id}' does not exist.")
            raise PineconeIndexNotFoundError(vectorstore_id)
        
        embeddings = OpenAIEmbeddings()
        vector_store = LangPinecone.from_existing_index(vectorstore_id, embeddings)
        logger.info(f"Pinecone vector store '{vectorstore_id}' loaded successfully.")
        return vector_store
    except PineconeIndexNotFoundError:
        raise
    except pinecone.PineconeException as e:
        logger.error(f"Pinecone service error while loading index '{vectorstore_id}': {e}", exc_info=True)
        raise PineconeServiceError("Failed to load Pinecone vector store.")
    except Exception as e:
        logger.exception(f"Unexpected error while loading vector store '{vectorstore_id}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load vector store.")

def create_qa_chain(vector_store, system_prompt: Optional[str] = None):
    """
    Create a CustomConversationalRetrievalChain with the provided vector store and optional system prompt.
    """
    try:
        # Initialize conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize Cerebras LLM with optional system prompt
        cerebras_llm = CerebrasLLM(client=client, model="llama3.1-8b")

        # Initialize custom RAG pipeline
        qa_chain = CustomConversationalRetrievalChain.from_llm(
            llm=cerebras_llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            system_prompt=system_prompt,
            verbose=True,
        )
        logger.info("Custom ConversationalRetrievalChain initialized successfully.")
        return qa_chain
    except Exception as e:
        logger.exception(f"Failed to create QA chain: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initialize QA chain.")

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

# =======================
# Initialize FastAPI App
# =======================

web_application = FastAPI(title="Extended Text to Pinecone Vector Store API")

# =======================
# Exception Handling
# =======================

@web_application.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@web_application.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error for request {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

@web_application.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )

# =======================
# API Endpoints
# =======================

# =======================
# Create Vector Store Endpoint
# =======================

@web_application.post("/create-vector-store", response_model=VectorStoreResponse)
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

# =======================
# Delete Vector Store Endpoint
# =======================

@web_application.post("/delete-vector-store", response_model=DeleteVectorStoreResponse)
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
# Start-Session Endpoint
# =======================

@web_application.post("/start-session", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Initializes a new chat session with a unique ID.
    """
    try:
        session_id = str(uuid.uuid4())
        memory = ConversationBufferMemory()
        memory.chat_memory.add_ai_message(Prompts.SYSTEM_PROMPT)
        sessions[session_id] = {
            "memory": memory
        }
        logger.info(f"Started new session with ID: {session_id}")
        return {"session_id": session_id}
    except Exception as e:
        logger.exception("Error starting new session")
        raise HTTPException(status_code=500, detail="Failed to start session")
    
# =======================
# Chat Endpoint
# =======================

@web_application.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Processes a chat message and returns an AI response.
    Optionally accepts a custom system prompt.
    """
    session_id = request.session_id
    user_message = request.message
    system_prompt = request.system_prompt  # Optional system prompt

    logger.debug(f"Received message for session {session_id}: {user_message}")

    if system_prompt:
        logger.debug(f"Using custom system prompt for session {session_id}: {system_prompt}")
    else:
        logger.debug(f"Using default system prompt for session {session_id}: {Prompts.SYSTEM_PROMPT}")

    # Validate session ID
    if session_id not in sessions:
        logger.warning(f"Session ID not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    memory = sessions[session_id]["memory"]

    try:
        # Add user message to memory
        memory.chat_memory.add_user_message(user_message)
        logger.debug(f"Added user message to session {session_id} memory")

        # Get conversation history
        history = memory.load_memory_variables({}).get("chat_history", "")
        logger.debug(f"Loaded history for session {session_id}: {history}")

        # Create prompt with history
        prompt = f"{history}\nUser: {user_message}\nAI:"
        logger.debug(f"Generated prompt for Cerebras: {prompt}")

        # Determine which system prompt to use
        current_system_prompt = system_prompt if system_prompt else Prompts.SYSTEM_PROMPT

        # Get AI response from Cerebras
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": current_system_prompt},
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
# RAG Chat Endpoint
# =======================

@web_application.post("/chat-rag", response_model=ChatRagResponse)
async def chat_rag_endpoint(request: ChatRagRequest):
    """
    Processes a chat message with RAG using a specified vector store and returns an AI response.
    Optionally accepts a custom system prompt.
    """
    session_id = request.session_id
    vectorstore_id = request.vectorstore_id
    user_message = request.message
    system_prompt = request.system_prompt  # Optional system prompt

    logger.debug(f"Received RAG message for session {session_id} with vectorstore_id {vectorstore_id}: {user_message}")

    # Validate session ID
    if session_id not in sessions:
        logger.warning(f"Session ID not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    memory = sessions[session_id]["memory"]

    try:
        # Load the specified vector store with retry
        vector_store = retry_operation(lambda: load_vectorstore(vectorstore_id))

        # Create QA chain with the loaded vector store and optional system prompt
        qa_chain = create_qa_chain(vector_store, system_prompt=system_prompt)

        # Generate AI response using the QA chain
        result = qa_chain({"question": user_message})
        ai_response = result["answer"]
        logger.debug(f"AI response for session {session_id} with RAG: {ai_response}")

        # Add AI response to memory
        memory.chat_memory.add_ai_message(ai_response)
        logger.debug(f"Added AI message to session {session_id} memory")

        return ChatRagResponse(
            session_id=session_id,
            vectorstore_id=vectorstore_id,
            input=user_message,
            response=ai_response
        )

    except PineconeIndexNotFoundError as e:
        raise e
    except PineconeServiceError as e:
        raise e
    except OpenAIServiceError as e:
        raise e
    except HTTPException as he:
        # Re-raise HTTP exceptions to be handled by FastAPI
        logger.error(f"HTTPException: {he.detail}", exc_info=True)
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error in chat_rag_endpoint for session {session_id}")
        raise HTTPException(status_code=500, detail="Failed to process chat message with RAG")
    
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
        
        return DeleteResponse(
            detail=f"Session '{session_id}' has been deleted successfully.",
            session_id=session_id,
            remaining_sessions=remaining
        )

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
# Application Startup and Shutdown Events
# =======================

@web_application.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("FastAPI application startup complete.")

@web_application.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("FastAPI application shutdown initiated.")

# =======================
# Modal Function to Run FastAPI Application
# =======================

@app.function(
    image=asgi_image,
    concurrency_limit=1, 
    allow_concurrent_inputs=500,
    secrets=[
        modal.Secret.from_name("OPENAI_API_KEY"),
        modal.Secret.from_name("PINECONE_API_KEY"),
        modal.Secret.from_name("PINECONE_ENVIRONMENT"),
        modal.Secret.from_name("CEREBRAS_API_KEY"),
        # Add other secrets if necessary
    ]
)
@modal.asgi_app(label="ExtendedFastAPIEndpoint")
def endpoint():
    return web_application

# =======================
# Main Entry Point for Local Testing
# =======================
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:web_application", host="0.0.0.0", port=8000, reload=True)