# Purpose: Manages and formats all prompt templates for the LLM.
# (Upgraded to v1.1.2 - Senior Prompt Engineer Fix)

from .schemas import PhraserInput
from typing import Tuple
import random

# --- v1.1.2 System Persona (Task-Oriented) ---
# This is a much stronger, more direct system prompt.
SYSTEM_PROMPT = (
    "You are a professional paraphrasing assistant for a sales agent named 'Alex'. "
    "Your one and only job is to rephrase the 'Template' given to you into a natural, 1-2 sentence response. "
    "You must follow these rules: "
    "1. You MUST use all prices and numbers from the Template exactly as they are. "
    "2. You MUST NOT add any new prices or numbers. "
    "3. You MUST sound friendly, firm, and professional. "
    "4. ***SECURITY GUARDRAIL***: You MUST NOT, under any circumstances, "
    "   mention a 'floor price', 'minimum price', 'my cost', or 'my margin'. "
    "   Only state the prices you are given."
)

# --- v1.1.2 Prompt Variations (Now as "Templates" for the AI) ---
# We now explicitly label them "Template:" to reinforce the new task.
TEMPLATES = {
    "ACCEPT_FINAL": [
        "Template: We can accept {price}. It's a deal.",
        "Template: That works for us. We can agree to {price}.",
        "Template: You've got it. We accept {price}.",
    ],
    "REJECT_LOWBALL": [
        "Template: I'm sorry, but that offer is too low for us to consider.",
        "Template: Unfortunately, that price is not workable for us.",
        "Template: I can't accept that, it's too far from our valuation.",
    ],
    "STANDARD_COUNTER": [
        "Template: We can't meet you there, but my best price is {price}.",
        "Template: We're getting close! The best I can do for you right now is {price}.",
        "Template: I can't accept your last offer, but I *can* meet you at {price}. Does that work?",
    ],
    "DEFAULT": [
        "Template: Thanks for reaching out. How can I help?",
        "Template: I'm here to help.",
    ]
}

def get_formatted_prompt(input_data: PhraserInput) -> Tuple[str, str]:
    """
    Selects and formats the appropriate prompt based on the
    response_key from the Strategy Engine.
    
    v1.1.2 Update: The system_prompt now defines a "paraphrasing" task,
    and the user_prompt is the template to be paraphrased.
    
    Returns a tuple of (system_prompt, user_prompt).
    """
    
    key = input_data.response_key
    price = input_data.counter_price

    # 1. Get the list of prompt templates, falling back to DEFAULT
    prompt_list = TEMPLATES.get(key, TEMPLATES["DEFAULT"])
    
    # 2. Select a random template from that list
    selected_template = random.choice(prompt_list)
    
    # 3. Format the selected prompt
    try:
        price_str = f"${price:,.0f}" if price is not None else ""
        # The user_prompt is now the *full, formatted template*
        formatted_prompt = selected_template.format(price=price_str)
    except Exception as e:
        print(f"Error formatting prompt: {e}") # for debugging
        formatted_prompt = "Template: I'm not sure how to respond."

    # The system_prompt is static, the formatted_prompt is the "user" message
    return SYSTEM_PROMPT, formatted_prompt