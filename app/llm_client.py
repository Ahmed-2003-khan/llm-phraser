# Purpose: Isolates all external LLM API logic using LangChain.
# Version: v2.0.0 - LangChain Integration

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from .schemas import PhraserInput
from .prompt_templates import get_prompt_template, format_price
import logging
import os

logger = logging.getLogger(__name__)

# Initialize the LangChain Chat Model
# Using the same model as before: llama-3.3-70b-versatile
# Lazy initialization to avoid errors during test collection
_llm = None

def get_llm():
    """Get or create the LangChain ChatGroq instance."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=128,
            api_key=os.environ.get("GROQ_API_KEY")
        )
    return _llm


async def generate_llm_response(input_data: PhraserInput) -> str:
    """
    Generates a persuasive response using LangChain and ChatGroq.
    
    This function:
    1. Selects the appropriate prompt template based on response_key
    2. Formats the counter_price (if present)
    3. Invokes the LangChain chain (prompt | llm | parser)
    4. Returns the generated text
    
    Args:
        input_data: PhraserInput containing action, response_key, and counter_price
        
    Returns:
        Generated response text as a string
        
    Raises:
        Exception: If LLM call fails (caught and returns fallback message)
    """
    
    try:
        # 1. Get the appropriate prompt template
        prompt_template = get_prompt_template(input_data)
        
        # 2. Format the price (if present)
        price_str = format_price(input_data.counter_price) if input_data.counter_price else ""
        
        logger.info(f"Generating phrase for key: {input_data.response_key}, price: {price_str}")
        
        # 3. Get the LLM instance
        llm = get_llm()
        
        # 4. Build the LangChain LCEL pipe
        output_parser = StrOutputParser()
        chain = prompt_template | llm | output_parser
        
        # 5. Invoke the chain asynchronously
        response_text = await chain.ainvoke({"price": price_str})
        
        # 6. Validate response
        if not response_text or response_text.strip() == "":
            logger.error("LLM returned an empty response.")
            return "I'm sorry, I'm not sure how to respond to that."
        
        logger.info(f"Generated response: {response_text}")
        return response_text.strip()
    
    except Exception as e:
        logger.error(f"Error calling LangChain/Groq API: {e}", exc_info=True)
        # Return a safe, generic fallback response
        return "We seem to be having a technical issue. Please try again in a moment."