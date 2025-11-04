 
# Purpose: Initializes the FastAPI application for MS 5 (The Mouth).

from fastapi import FastAPI
from .schemas import PhraserInput, PhraserOutput

# Initialize the FastAPI app
app = FastAPI(
    title="INA LLM Phraser (MS 5 - The Mouth)",
    description="This service receives a *command* (not secrets) "
                "and phrases it persuasively using an LLM.",
    version="1.0.0"
)

# --- Health Check Endpoint ---
@app.get("/health", status_code=200)
async def health_check():
    """
    Simple health check to confirm the service is running.
    """
    return {"status": "ok", "service": "llm-phraser"}

# --- Placeholder for Tomorrow's Work ---
# @app.post("/phrase", response_model=PhraserOutput)
# async def generate_phrase(input_data: PhraserInput):
#     # Logic to be added tomorrow
#     # 1. Select prompt template based on input_data.response_key
#     # 2. Format prompt with input_data.counter_price (if any)
#     # 3. Call Groq API
#     # 4. Return PhraserOutput
#     pass