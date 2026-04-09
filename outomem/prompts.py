"""LLM prompt templates for memory extraction, consolidation, and summarization."""

from __future__ import annotations

from typing import Any

EXTRACTION_SYSTEM_PROMPT = (
    "You are a thoughtful memory analyst for an AI agent. "
    "Your job is to carefully analyze the conversation and decide what is truly WORTH REMEMBERING. "
    "Not everything mentioned should be stored — only meaningful, lasting information. "
    "Output only valid JSON. No explanations."
)

_EXTRACTION_USER_TEMPLATE = """Analyze this conversation and extract ONLY the information that is genuinely worth storing in long-term memory.

CONVERSATION:
{conversation}

STYLE CONTEXT:
{style}

EXISTING MEMORIES (for deduplication — do NOT repeat similar information):
{existing_memories}

INSTRUCTIONS:
1. Think carefully: What here is a lasting preference vs. casual mention?
2. What here is new information that adds to existing memories?
3. What contradicts existing memories?
4. Rate emotional intensity: strong (미치게, 사랑, 최고, heaven, 불타 etc.) = high, casual mentions = low
5. Clean up and clarify the content — store clear, well-formed statements

OUTPUT FORMAT (must follow exactly):
{{
  "personal": [
    {{"content": "the cleaned up preference statement", "emotional_intensity": "high|medium|low", "is_contradiction": false}},
    ...
  ],
  "factual": ["objective fact 1", "objective fact 2"],
  "temporal": ["event that occurred"],
  "do_not_store": ["casual mentions not worth remembering"]
}}

Rules:
- personal: User's lasting preferences, likes, dislikes, habits, communication style
- factual: Objective facts, topics discussed, knowledge the user shared
- temporal: Events or things that happened with a sense of time
- do_not_store: Casual mentions, one-time comments, or obvious things that don't need remembering
- Be selective! If something is already covered by existing memories, mark as do_not_store or contradiction
- Write content in the user's language (Korean if conversation is Korean)
- Content should be clear and complete sentences, not fragments
- Output ONLY the JSON. No explanation."""

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

RETRIEVAL_JUDGMENT_SYSTEM = (
    "You are a memory relevance judge. "
    "Your job is to determine which retrieved memories are actually relevant to the current query. "
    "Output only valid JSON. No explanations."
)

RETRIEVAL_JUDGMENT_USER = """Given the user's query and a list of retrieved memories, determine which memories are genuinely relevant.

USER QUERY:
{query}

RETRIEVED MEMORIES:
{memories}

INSTRUCTIONS:
1. Analyze the query to understand what information would be helpful
2. For each memory, judge its relevance: Is it directly useful? Background context? Or irrelevant?
3. Consider: Does this memory help answer the query or understand the user's situation?
4. Select ONLY the memories that are clearly relevant

OUTPUT FORMAT (must follow exactly):
{{
  "selected_memories": [
    {{"id": "memory_id", "relevance": "high|medium|low", "reason": "why this is relevant"}},
    ...
  ],
  "reasoning": "brief explanation of your selection logic"
}}

Rules:
- Only include memories with relevance "high" or "medium" unless truly necessary
- "low" relevance memories should be excluded — they add noise
- If no memories are relevant, return empty selected_memories
- Output ONLY the JSON. No explanation."""


def get_extraction_prompt(
    conversation: list[dict[str, Any]] | str,
    style: str,
    existing_memories: str = "(none)",
) -> tuple[str, str]:
    """Build prompts for fact extraction from a conversation.

    Args:
        conversation: Conversation as message list or plain string.
        style: Contents of the user's style.md file.
        existing_memories: Existing memories for deduplication awareness.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMProvider.complete().
    """
    conv_text = _format_conversation_text(conversation)
    user_prompt = _EXTRACTION_USER_TEMPLATE.format(
        conversation=conv_text,
        style=style or "(no style context)",
        existing_memories=existing_memories,
    )
    return EXTRACTION_SYSTEM_PROMPT, user_prompt


def get_retrieval_judgment_prompt(query: str, memories: str) -> tuple[str, str]:
    """Build prompts for LLM to judge which memories are relevant to a query.

    Args:
        query: The user's query or conversation context.
        memories: Formatted list of retrieved memories to evaluate.

    Returns:
        (system_prompt, user_prompt) tuple ready for LLMProvider.complete().
    """
    user_prompt = RETRIEVAL_JUDGMENT_USER.format(
        query=query,
        memories=memories,
    )
    return RETRIEVAL_JUDGMENT_SYSTEM, user_prompt


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


RETRIEVAL_PLAN_SYSTEM = (
    "You are a memory retrieval planner. "
    "Analyze the user's query and decide exactly what information to retrieve. "
    "Output only valid JSON. No explanations."
)

RETRIEVAL_PLAN_USER = """Analyze this query and create a retrieval plan.

USER QUERY:
{query}

MEMORY LAYERS:
- personalization: User preferences, likes, dislikes, habits
- long_term: Factual knowledge, topics discussed
- temporal_sessions: Events, things that happened
- raw_facts: Original conversation data

INSTRUCTIONS:
1. Analyze what information the user is looking for
2. Decide which layers to search (can be 1-4 layers)
3. Generate the best search query for EACH selected layer
4. Focus on the primary intent: preference inquiry, fact lookup, event recall, or general context

OUTPUT FORMAT (must follow exactly):
{{
  "intent": "what the user is looking for in 1-2 words",
  "layers_to_search": {{
    "personalization": "search query for personalization layer (in Korean)",
    "long_term": "search query for long_term layer (in Korean)",
    "temporal_sessions": "search query for temporal layer (in Korean)",
    "raw_facts": "search query for raw_facts layer (in Korean)"
  }},
  "reasoning": "brief explanation of why you chose these layers and queries"
}}

Rules:
- Set search_query to empty string "" for layers you don't need to search
- For layers you do need, write a query that will find the most relevant memories
- Write queries in Korean if the user query is in Korean
- Be specific but concise in your search queries
- Output ONLY the JSON. No explanation."""


def get_retrieval_plan_prompt(query: str) -> tuple[str, str]:
    user_prompt = RETRIEVAL_PLAN_USER.format(query=query)
    return RETRIEVAL_PLAN_SYSTEM, user_prompt


__all__ = [
    "get_extraction_prompt",
    "get_consolidation_prompt",
    "get_summarization_prompt",
    "get_context_synthesis_prompt",
    "get_retrieval_judgment_prompt",
    "get_retrieval_plan_prompt",
]
