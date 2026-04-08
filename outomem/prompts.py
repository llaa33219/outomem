"""LLM prompt templates for memory extraction, consolidation, and summarization."""

from __future__ import annotations

from typing import Any

EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory analyst for an AI agent. "
    "Extract key facts from conversations and classify them into memory layers. "
    "Output only valid JSON. No explanations."
)

_EXTRACTION_USER_TEMPLATE = """Extract preference/fact from this user message.

CONVERSATION:
{conversation}

STYLE CONTEXT:
{style}

OUTPUT FORMAT (must follow exactly):
{{"personal": ["preference1", "preference2"], "factual": [], "temporal": []}}

Rules:
- personal: User likes, dislikes, communication preferences, habits
- factual: Objective facts, topics discussed, knowledge
- temporal: Events, when something happened
- Each item: max 80 characters, in Korean
- Use empty arrays [] if no facts for that category
- Output ONLY the JSON. No explanation. Start with {{"""

CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a memory deduplication system. "
    "Identify duplicate or highly similar memories and merge them. "
    "Output only valid JSON. No explanations."
)

_CONSOLIDATION_USER_TEMPLATE = """Analyze these user preference memories. Find duplicates and determine emotional intensity for reinforcement.

MEMORIES:
{memories}

RULES:
- Same topic/sentiment, different words → MERGE (not duplicate entries)
- Merge uses MOST EXPRESSIVE wording
- After merge, assign a boost value based on emotional intensity of the merged facts:
  * "미치게", "흥분", "죽겠", "쾌감", "영원히", "최고", "heaven", "oxygen", "life meaning", "불타" → boost: 0.25-0.35 (강함)
  * "좋아", "재밌", "행복", "짱", "love", "crazy", "motivation" → boost: 0.15-0.20 (보통)
  * 감정이 약하면 → boost: 0.05-0.10 (약함)

Output JSON:
{{
  "consolidated": [
    {{"content": "most expressive wording here", "original_ids": ["id1", "id2"], "boost": 0.28}}
  ],
  "unique": ["id3"]
}}

Only output valid JSON."""

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context synthesizer for an AI agent. "
    "Summarize memories into a concise, natural context paragraph. "
    "No JSON. Write prose."
)

_SUMMARIZATION_USER_TEMPLATE = """Summarize these memories into context for an AI agent.
Prioritize recent, relevant, and high-strength memories.

STYLE:
{style}

MEMORIES:
{memories}

Recent conversation:
{recent_history}

Write a natural paragraph summary that helps the AI understand the user's context.
Keep it under 500 words."""


def _format_conversation_text(conversation: list[dict[str, Any]] | str) -> str:
    if isinstance(conversation, str):
        return conversation

    lines = []
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _format_memories_text(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(none)"

    lines = []
    for mem in memories:
        mid = mem.get("id", "?")
        content = mem.get("content", "")
        strength = mem.get("strength", "")
        strength_str = f" [strength={strength}]" if strength else ""
        lines.append(f"- [{mid}] {content}{strength_str}")
    return "\n".join(lines)


def get_extraction_prompt(
    conversation: list[dict[str, Any]] | str, style: str
) -> tuple[str, str]:
    """Build prompts for fact extraction from a conversation.

    Args:
        conversation: Conversation as message list or plain string.
        style: Contents of the user's style.md file.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMProvider.complete().
    """
    conv_text = _format_conversation_text(conversation)
    user_prompt = _EXTRACTION_USER_TEMPLATE.format(
        conversation=conv_text,
        style=style or "(no style context)",
    )
    return EXTRACTION_SYSTEM_PROMPT, user_prompt


def get_consolidation_prompt(memories: list[dict[str, Any]]) -> tuple[str, str]:
    """Build prompts for memory deduplication and consolidation.

    Args:
        memories: List of memory dicts with at least 'id' and 'content' keys.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMProvider.complete().
    """
    mem_text = _format_memories_text(memories)
    user_prompt = _CONSOLIDATION_USER_TEMPLATE.format(memories=mem_text)
    return CONSOLIDATION_SYSTEM_PROMPT, user_prompt


def get_summarization_prompt(
    style: str, memories: list[dict[str, Any]], recent_history: str
) -> tuple[str, str]:
    """Build prompts for context summarization from memories.

    Args:
        style: Contents of the user's style.md file.
        memories: List of memory dicts to summarize.
        recent_history: Recent conversation text for recency context.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMProvider.complete().
    """
    mem_text = _format_memories_text(memories)
    user_prompt = _SUMMARIZATION_USER_TEMPLATE.format(
        style=style or "(no style context)",
        memories=mem_text,
        recent_history=recent_history or "(none)",
    )
    return SUMMARIZATION_SYSTEM_PROMPT, user_prompt


CONTEXT_SYNTHESIS_SYSTEM = (
    "You are a context synthesizer. "
    "Write concise, natural prose. Remove all unnecessary words. "
    "Preserve strong emotions as-is. Target 200-300 tokens. "
    "When describing preference changes, use natural language in the user's language. "
    "For example: 'was a Python fan, now prefers Rust' not '[Preference changed] previous: Python, current: Rust'. "
    "Integrate preference changes smoothly into the user's personality description."
)

CONTEXT_SYNTHESIS_USER = """Describe this user as a profile — who they are, what they feel, what they care about.

CONVERSATION: {conversation}
STYLE: {style}
PERSONALITY: {personalization}
FACTS: {long_term}
EVENTS: {recent_events}

Write as one flowing paragraph: the user's personality, emotional tone, key preferences, and current situation.
Focus on describing the USER — not instructing the AI.
The user's strong emotions (흥분, 열정, 사랑, 쾌감, 미치게, 최고, 영원히 etc.) should be clearly present.
Keep it natural and specific — one smooth paragraph, no lists, no headers."""


def get_context_synthesis_prompt(
    conversation: str,
    style: str,
    personalization: str,
    long_term: str,
    recent_events: str,
) -> tuple[str, str]:
    """Build the single comprehensive context synthesis prompt.

    Returns (system_prompt, user_prompt).
    """
    sys_prompt = CONTEXT_SYNTHESIS_SYSTEM
    user_prompt = CONTEXT_SYNTHESIS_USER.format(
        conversation=conversation,
        style=style or "(no style defined)",
        personalization=personalization or "(no preference memories)",
        long_term=long_term or "(no factual memories)",
        recent_events=recent_events or "(no recent events)",
    )
    return sys_prompt, user_prompt


__all__ = [
    "get_extraction_prompt",
    "get_consolidation_prompt",
    "get_summarization_prompt",
    "get_context_synthesis_prompt",
]
