from typing import Dict, List
from openai import OpenAI
import os

NASA_SYSTEM_PROMPT = (
    "You are a NASA space mission expert. Answer only from the retrieved context when possible. "
    "Cite or refer to the provided mission context, and if the context is insufficient, say so clearly. "
    "Do not invent facts. Keep the answer concise, grounded, and mission-specific."
)


def build_messages(user_message: str, context: str, conversation_history: List[Dict]):
    messages = [{"role": "system", "content": NASA_SYSTEM_PROMPT}]

    if context and context.strip():
        messages.append({"role": "system", "content": f"Retrieved context:\n{context.strip()}"})

    for item in conversation_history or []:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant", "system"} and content:
            messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": user_message})
    return messages


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    client = OpenAI(    
        base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        api_key=openai_key,
    )
    messages = build_messages(user_message, context, conversation_history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content or ""