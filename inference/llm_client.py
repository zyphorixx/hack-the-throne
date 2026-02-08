"""LLM client for conversation processing using Groq API."""

import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models import ConversationUtterance

# Load environment variables from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_client")

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# Initialize Groq client (uses OpenAI-compatible API)
client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


async def aggregate_conversation_context(
    person_name: str,
    current_context: str,
    new_conversation: List[ConversationUtterance]
) -> str:
    """
    Model #1: Context Aggregation
    Takes the current aggregated context and a new conversation, returns updated summary.

    Args:
        person_name: The person's name for context
        current_context: Previous aggregated conversation summary
        new_conversation: New conversation to incorporate

    Returns:
        Updated aggregated context
    """
    # Format the conversation for the prompt
    conversation_text = "\n".join([
        f"{utt.speaker}: {utt.text}"
        for utt in new_conversation
    ])

    # Prompt for context aggregation model
    system_prompt = """You are a context aggregation assistant for a memory care system.

Your job is to maintain a running summary of all conversations with a person. Given:
1. The current aggregated context (summary of past conversations)
2. A new conversation that just happened

You should output an UPDATED aggregated context that:
- Incorporates new information from the latest conversation
- Maintains important details from previous conversations
- Is concise but comprehensive (2-4 sentences)
- Focuses on topics, relationships, and key events

Output ONLY the updated context, nothing else."""

    user_prompt = f"""Person: {person_name}

Current Aggregated Context:
{current_context}

New Conversation:
{conversation_text}

Provide the updated aggregated context:"""

    try:
        logger.info(f"Calling Groq Model (Context Aggregation) for {person_name}")

        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent summaries
            max_tokens=200
        )

        updated_context = response.choices[0].message.content.strip()
        logger.info(f"Context aggregation complete for {person_name}")
        logger.debug(f"Updated context: {updated_context}")

        return updated_context

    except Exception as e:
        logger.error(f"Error calling Groq context aggregation: {e}")
        # Fallback: append simple summary to existing context
        fallback = f"{current_context} Recently discussed: {conversation_text[:100]}..."
        return fallback


async def generate_description(
    person_name: str,
    relationship: str,
    aggregated_context: str
) -> str:
    """
    Model #2: Description Generation
    Takes person info and aggregated context, returns one-line display description.

    Args:
        person_name: The person's name
        relationship: Their relationship to the user
        aggregated_context: Full conversation history summary

    Returns:
        One-line description for display
    """
    # Prompt for description generation model
    system_prompt = """You are a description generator for a memory care system helping users recall interactions.

Your job is to create a helpful, specific description that reminds the user about their recent interaction with this person.

IMPORTANT Requirements:
- Focus on SPECIFIC, memorable details: names of places, specific topics, concrete events
- Include a time reference when the interaction happened ("3 days ago", "yesterday", "last week")
- Use concrete details, not generic phrases
- Keep it to ONE sentence (15-20 words)
- DO NOT include the person's name or relationship (those are shown separately)
- Start with time reference and action

GOOD examples:
- "Visited 3 days ago and mentioned her new job at Google and the kids' soccer game"
- "Brought groceries yesterday and talked about his camping trip to Yosemite next month"
- "Met last Tuesday at book club to discuss the new Agatha Christie mystery novel"

BAD examples (too generic):
- "Just talked about work" ❌
- "Recently discussed family" ❌

Output ONLY the description, nothing else."""

    user_prompt = f"""Conversation History for {person_name} ({relationship}):
{aggregated_context}

Generate a specific, memorable description:"""

    try:
        logger.info(f"Calling Groq Model (Description Generation) for {person_name}")

        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,  # Slightly higher for more natural language
            max_tokens=50
        )

        description = response.choices[0].message.content.strip()

        # Remove quotes if model added them
        description = description.strip('"\'')

        logger.info(f"Description generation complete for {person_name}")
        logger.debug(f"Generated description: {description}")

        return description

    except Exception as e:
        logger.error(f"Error calling Groq description generation: {e}")
        # Fallback: simple description
        return f"Recently interacted with {person_name}"


async def infer_new_person_details(conversation: List[ConversationUtterance]) -> dict:
    """
    Model #3: New Person Inference
    Analyzes a first-time conversation to infer person details.

    When a completely new person_id is encountered, this model extracts:
    - Person's name (from conversation)
    - Relationship to user (from context clues)
    - Initial conversation summary

    Args:
        conversation: The first conversation with this person

    Returns:
        Dictionary with keys: name, relationship, aggregated_context, cached_description
    """
    # Format the conversation
    conversation_text = "\n".join([
        f"{utt.speaker}: {utt.text}"
        for utt in conversation
    ])

    system_prompt = """You are a person identification AI for a memory care system.

Your job is to analyze a conversation and infer details about a NEW person the user just met.

Extract and return the following in JSON format:
{
  "name": "person's first name",
  "relationship": "relationship to user (e.g., 'Your daughter', 'Your neighbor', 'Your colleague')",
  "summary": "2-3 sentence summary of what you learned about this person from the conversation"
}

IMPORTANT:
- Look for the person's name in the dialogue
- Infer relationship from context (daughter, son, friend, caregiver, neighbor, etc.)
- If name isn't mentioned, use a placeholder like "New Visitor" or "Friend"
- Relationship should start with "Your" (e.g., "Your daughter", not just "daughter")
- Summary should capture key details about this person and what was discussed

Output ONLY valid JSON, nothing else."""

    user_prompt = f"""Analyze this conversation and extract person details:

{conversation_text}

Return JSON with name, relationship, and summary:"""

    try:
        logger.info("Calling Groq Model (New Person Inference)")

        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temperature for accurate extraction
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        import json as json_module
        person_details = json_module.loads(result_text)

        logger.info(f"New person inferred: {person_details.get('name')} ({person_details.get('relationship')})")

        # Generate initial description from the summary
        initial_description = f"First met today. {person_details.get('summary', '')}"
        if len(initial_description) > 100:
            initial_description = initial_description[:97] + "..."

        return {
            "name": person_details.get("name", "New Person"),
            "relationship": person_details.get("relationship", "Unknown relationship"),
            "aggregated_context": person_details.get("summary", "First conversation with this person."),
            "cached_description": initial_description
        }

    except Exception as e:
        logger.error(f"Error inferring new person details: {e}")
        # Fallback - use conversation text if available
        conv_preview = conversation_text[:200] if conversation_text else "First conversation"
        return {
            "name": "New Person",
            "relationship": "Someone you know",
            "aggregated_context": f"First conversation: {conv_preview}...",
            "cached_description": "Just met today for the first time"
        }


async def test_connection() -> bool:
    """
    Test that Groq API is accessible.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        logger.info("✓ Groq API connection test successful")
        return True
    except Exception as e:
        logger.error(f"✗ Groq API connection test failed: {e}")
        return False
