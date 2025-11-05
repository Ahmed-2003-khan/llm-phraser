# Purpose: Manages and formats all prompt templates for the LLM.
# This is the "leash" that controls the LLM's responses.

from .schemas import PhraserInput
from typing import Tuple

def get_formatted_prompt(input_data: PhraserInput) -> Tuple[str, str]:
    """
    Selects and formats the appropriate prompt based on the
    response_key from the Strategy Engine.
    
    Returns a tuple of (system_prompt, user_prompt).
    """
    
    key = input_data.response_key
    price = input_data.counter_price

    # Define the persona for the bot
    # This is a "system prompt" that applies to all responses.
    system_prompt = (
        "You are 'Alex', a helpful and professional customer support agent "
        "for a premium electronics store. You are friendly, firm, and "
        "persuasive. Your goal is to close the sale. "
        "Do NOT use emojis. Do NOT sound like a robot. "
        "Keep your responses to 1-2 sentences."
    )

    # Define the specific task templates
    templates = {
        "ACCEPT_FINAL": (
            "The user's offer is great. Enthusiastically accept the offer "
            f"of ${price:,.0f} and tell them you're preparing the item."
        ),
        "REJECT_LOWBALL": (
            "The user's offer is far too low. Gently but firmly reject it. "
            "Do not provide a counter-offer. "
            "Explain that the offer is too far from the item's value."
        ),
        "STANDARD_COUNTER": (
            "The user's offer is reasonable, but not high enough. "
            "Politely reject their last offer and propose your new counter-offer "
            f"of **${price:,.0f}**. Briefly explain it's a fair price."
        ),
        # --- Add more keys as your 'Brain' (MS 4) evolves ---
        
        "DEFAULT": (
            "The user has said something. Respond politely. "
            "If they made an offer, be vague and ask them to wait."
        )
    }

    # Select the template, using a fallback for safety
    user_prompt = templates.get(key, templates["DEFAULT"])

    return system_prompt, user_prompt