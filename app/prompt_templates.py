# Purpose: Manages and formats all prompt templates for the LLM using LangChain.
# Version: v2.1.0 - Multi-Example Prompt Support

from langchain_core.prompts import ChatPromptTemplate
from .schemas import PhraserInput
from typing import Dict, List
import logging
import random

logger = logging.getLogger(__name__)

# --- Security-Hardened System Prompt ---
SYSTEM_PROMPT = (
    "You are a professional paraphrasing assistant for a sales agent named 'Alex'. "
    "Your job is to convert instructions or example templates into a single, natural, conversational response. "
    "\n\n**CRITICAL RULES:**\n"
    "1. You MUST use all prices and numbers exactly as provided in the instruction.\n"
    "2. You MUST NOT invent, hallucinate, or change any prices or numbers.\n"
    "3. Your responses must be 1-2 sentences maximum.\n"
    "4. Sound friendly, firm, and professional.\n"
    "5. **SECURITY GUARDRAIL**: You MUST NOT mention 'floor price', 'minimum price', "
    "'my cost', 'my margin', or any internal financial metrics. Only state the prices given to you."
)

# --- Raw Template Variations ---
# We define potential variations for each intent.
# These will be presented to the LLM as "Examples of what we want to say".

TEMPLATE_VARIATIONS: Dict[str, List[str]] = {
    
    "ACCEPT_FINAL": [
        "We can accept {price}. It's a deal.",
        "That works for us. We can agree to {price}.",
        "You've got it. We accept {price}. Let's finalize this."
    ],
    
    "ACCEPT_SENTIMENT_CLOSE": [
        "Alright, I can see this is important to you. We'll accept {price} to move forward.",
        "I understand your position. Ideally we'd want more, but we can do {price}.",
        "To get this done today, I'll agree to {price}."
    ],
    
    "REJECT_LOWBALL": [
        "I appreciate the offer, but that price is too low for us to consider. Do NOT propose a counter-offer.",
        "That offer doesn't work for us unfortunately. It's below our range. Do NOT counter.",
        "We can't get there. That price is just too low. Do NOT suggest a new price."
    ],
    
    "STANDARD_COUNTER": [
        "We're getting close! I can't meet you at your last offer, but my best price right now is {price}.",
        "I can't do that, but I can come down to {price}. How does that sound?",
        "We have a bit of a gap. I can meet you halfway at {price}."
    ],
    
    "COUNTER_FINAL_OFFER": [
        "This is my absolute final offer: {price}. I cannot go any lower.",
        "I've stretched as far as I can. {price} is my limit.",
        "To be transparent, {price} is the absolute best I can do."
    ],
    
    "DEFAULT": [
        "Thanks for reaching out. How can I help you today?",
        "I'm here to help. What's on your mind?",
    ]
}


def get_prompt_template(input_data: PhraserInput) -> ChatPromptTemplate:
    """
    Constructs a LangChain ChatPromptTemplate that includes multiple examples
    for the given response_key.
    
    Args:
        input_data: The PhraserInput containing the response_key
        
    Returns:
        A ChatPromptTemplate instance ready to be invoked
    """
    key = input_data.response_key
    
    # Get the list of variations, defaulting if key not found
    variations = TEMPLATE_VARIATIONS.get(key, TEMPLATE_VARIATIONS["DEFAULT"])
    
    # We join all variations into a single string to show the LLM the "Vibe/Range"
    # or we could select one randomly. The user requested "keep multiple response strings...
    # so the LLM can generate responses based on them".
    #
    # Strategy: Present all variations as a list of "Reference Styles".
    
    formatted_variations = "\n".join([f"- {v}" for v in variations])
    
    user_message_content = (
        f"Here are varying examples of how we want to respond to the customer:\n\n"
        f"{formatted_variations}\n\n"
        f"Task: Generate a single, natural response that captures the intent of these examples. "
        f"Paraphrase it naturally."
    )
    
    logger.info(f"Selected {len(variations)} variations for key: {key}")
    
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", user_message_content)
    ])


def format_price(price: float) -> str:
    """
    Formats a price value as a currency string.
    """
    if price is None:
        return ""
    return f"${price:,.0f}"