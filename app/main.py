# Purpose: Initializes the FastAPI application and defines API endpoints.
# Version: v2.0.0 - LangChain Integration (Simplified)

from fastapi import FastAPI, HTTPException
from .schemas import PhraserInput, PhraserOutput
from .llm_client import generate_llm_response

import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    Note: With LangChain, we don't need to manually manage the Groq client.
    ChatGroq manages its own HTTP client internally.
    """
    logger.info("🚀 LLM Phraser (MS 5) starting up...")
    logger.info("LangChain ChatGroq initialized.")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="INA LLM Phraser (MS 5 - The Mouth)",
    description="This service receives a *command* (not secrets) "
                "and phrases it persuasively using an LLM via LangChain.",
    version="2.0.0",
    lifespan=lifespan
)


# --- Health Check Endpoint ---
@app.get("/health", status_code=200)
async def health_check():
    """
    Health check endpoint to verify service is running.
    """
    return {"status": "ok", "service": "llm-phraser", "version": "2.0.0"}


# --- LLM Phrasing Endpoint (LangChain-powered) ---
@app.post("/phrase", response_model=PhraserOutput)
async def generate_phrase(input_data: PhraserInput):
    """
    Receives a command from the Strategy Engine (MS 4) and
    generates a persuasive, natural language response using LangChain.
    
    Security Note:
    - This endpoint NEVER receives the 'mam' (Minimum Acceptable Margin).
    - The PhraserInput schema enforces this firewall.
    - The LLM is sandboxed from sensitive financial data.
    
    Args:
        input_data: PhraserInput containing action, response_key, and optional counter_price
        
    Returns:
        PhraserOutput containing the generated response_text
    """
    
    try:
        # Call the LangChain-powered generation function
        # No need to inject a client - ChatGroq is initialized in llm_client.py
        response_text = await generate_llm_response(input_data)
        
        # Return the response
        return PhraserOutput(response_text=response_text)

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unhandled error in /phrase endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An internal server error occurred."
        )