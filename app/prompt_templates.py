# Purpose: Manages and formats all prompt templates for the LLM using LangChain.
# Version: v2.0.0 - LangChain Integration with Contextual Prompts

from langchain_core.prompts import ChatPromptTemplate
from .schemas import PhraserInput
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# --- Security-Hardened System Prompt ---
SYSTEM_PROMPT = (
    "You are a professional paraphrasing assistant for a sales agent named 'Alex'. "
    "Your job is to convert instructions into natural, conversational responses. "
    "\n\n**CRITICAL RULES:**\n"
    "1. You MUST use all prices and numbers exactly as provided in the instruction.\n"
    "2. You MUST NOT invent, hallucinate, or change any prices or numbers.\n"
    "3. Your responses must be 1-2 sentences maximum.\n"
    "4. Sound friendly, firm, and professional.\n"
    "5. **SECURITY GUARDRAIL**: You MUST NOT mention 'floor price', 'minimum price', "
    "'my cost', 'my margin', or any internal financial metrics. Only state the prices given to you."
)

# --- Contextual Prompt Templates (LangChain) ---
# Each template is mapped to a response_key from the Strategy Engine

PROMPT_TEMPLATES: Dict[str, ChatPromptTemplate] = {
    
    # Professional, clear, deal-closing acceptance
    "ACCEPT_FINAL": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: We can accept {price}. It's a deal. Let's finalize this."
        )
    ]),
    
    # Reluctant acceptance (used when accepting due to user frustration)
    "ACCEPT_SENTIMENT_CLOSE": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: Alright, I can see this is important to you. "
         "We'll accept {price} to move forward. You've got a deal."
        )
    ]),
    
    # Firm, polite rejection WITHOUT counter-offer
    # This is critical - the LLM must NOT generate a new price
    "REJECT_LOWBALL": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: I appreciate the offer, but that price is too low for us to consider. "
         "We cannot proceed at that level. Do NOT propose a counter-offer."
        )
    ]),
    
    # Encouraging counter-offer ("we're getting close")
    "STANDARD_COUNTER": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: We're getting close! I can't meet you at your last offer, "
         "but my best price right now is {price}. How does that sound?"
        )
    ]),
    
    # Firm, conclusive final offer
    "COUNTER_FINAL_OFFER": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: This is my absolute final offer: {price}. "
         "This is the best I can do, and I cannot go any lower. "
         "This is my limit."
        )
    ]),
    
    # Fallback for unknown keys
    "DEFAULT": ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", 
         "Convert this into a natural response:\n"
         "Template: Thanks for reaching out. How can I help you today?"
        )
    ])
}


def get_prompt_template(input_data: PhraserInput) -> ChatPromptTemplate:
    """
    Selects the appropriate LangChain ChatPromptTemplate based on 
    the response_key from the Strategy Engine.
    
    Args:
        input_data: The PhraserInput containing the response_key
        
    Returns:
        A ChatPromptTemplate instance ready to be invoked
    """
    key = input_data.response_key
    
    # Get the template, defaulting to DEFAULT if key not found
    template = PROMPT_TEMPLATES.get(key, PROMPT_TEMPLATES["DEFAULT"])
    
    logger.info(f"Selected prompt template for key: {key}")
    
    return template


def format_price(price: float) -> str:
    """
    Formats a price value as a currency string.
    
    Args:
        price: The price value to format
        
    Returns:
        Formatted price string (e.g., "$48,000")
    """
    if price is None:
        return ""
    return f"${price:,.0f}"