# Context Synthesis Guide

Context synthesis is the process of transforming raw, multi-layered memories into a coherent, natural-language profile of the user. This guide explains how Outomem builds this context to help AI agents understand who they are interacting with.

## What Context Synthesis Does

Context synthesis turns stored memories—preferences, facts, and recent events—into a single, flowing natural-language paragraph. Instead of providing the LLM with a dry list of facts, Outomem creates a "personality profile" that captures the user's emotional tone, current situation, and core values.

## The Pipeline

The `get_context()` method follows a 11-step pipeline to generate the final context string:

1.  **Conversation Formatting**: The recent conversation history is formatted into a plain text block.
2.  **Embedding Generation**: A query embedding is computed from the formatted conversation text.
3.  **Vector Search**: The system performs a semantic search across four LanceDB layers (personalization, long_term, raw_facts, and temporal_sessions).
4.  **Distance Sorting**: Results from all layers are sorted by their vector distance to ensure relevance.
5.  **Strength Recalculation**: Memory decay is applied to all stored facts to prioritize fresh or reinforced information.
6.  **Access Recording**: The system records an "access event" for retrieved personalization memories in both LanceDB and Neo4j.
7.  **Memory Formatting**: Retrieved memories are converted into structured text lists (with strength indicators for personalization).
8.  **Empty Check**: If no memories are found, the system returns the base style context or an empty string.
9.  **LLM Synthesis**: The system sends the formatted memories and conversation context to an LLM using the synthesis prompt.
10. **Fallback Execution**: If the LLM call fails, the system generates a structured Korean-language summary as a fallback.
11. **Token Truncation**: The final output is truncated to fit within the specified `max_tokens` limit.

## Search Strategy

Outomem searches specific layers with predefined limits to balance breadth and performance:

*   **personalization**: limit 5 (User preferences and habits)
*   **long_term**: limit 5 (Objective facts and knowledge)
*   **raw_facts**: limit 2 (Unprocessed or low-level facts)
*   **temporal_sessions**: limit 5 (Recent events and session-specific context)

## Strength Recalculation

Memory strength is not static. Before synthesis begins, `get_context()` triggers `_recalculate_strengths()`. This applies a decay function to all memories, ensuring that older, unreinforced memories lose influence while frequently accessed or emotionally charged memories remain prominent in the synthesized output.

## The Synthesis Prompt

The system uses a two-part prompt to guide the LLM in creating the user profile.

### System Prompt
```text
You are a context synthesizer. Write concise, natural prose. Remove all unnecessary words. 
Preserve strong emotions as-is. Target 200-300 tokens. 
When describing preference changes, use natural language in the user's language. 
For example: 'was a Python fan, now prefers Rust' not '[Preference changed] previous: Python, current: Rust'. 
Integrate preference changes smoothly into the user's personality description.
```

### User Prompt Template
```text
Describe this user as a profile — who they are, what they feel, what they care about.

CONVERSATION: {conversation}
STYLE: {style}
PERSONALITY: {personalization}
FACTS: {long_term}
EVENTS: {recent_events}

Write as one flowing paragraph: the user's personality, emotional tone, key preferences, and current situation.
Focus on describing the USER — not instructing the AI.
The user's strong emotions (흥분, 열정, 사랑, 쾌감, 미치게, 최고, 영원히 etc.) should be clearly present.
Keep it natural and specific — one smooth paragraph, no lists, no headers.
```

## Fallback Behavior

If the LLM synthesis fails (e.g., due to API timeout or rate limits), Outomem falls back to a hardcoded Korean format to ensure the agent still receives critical context:

`사용자 취향: {pers}. 핵심 사실: {lt}. 최근 사건: {temp}. 참고: {raw}`

Each section is joined by a period and space. If a specific layer has no data, it is omitted from the fallback string.

## Token Truncation

To prevent context window overflow, the system uses `truncate_to_token_limit()`. This utility:
1.  Counts tokens using the model's specific tokenizer (e.g., tiktoken for OpenAI).
2.  If the limit is exceeded, it attempts to cut the text at the nearest sentence boundary (searching for the last period) to maintain readability.
3.  Ensures the final string is strictly within the `max_tokens` budget.

## Example Output

**Input Memories:**
*   Personalization: "Likes dark mode", "Hates small fonts"
*   Long Term: "Software engineer", "Lives in Seoul"
*   Recent Events: "Just finished a long debugging session"

**Synthesized Paragraph:**
The user is a Seoul-based software engineer who deeply values visual comfort, strictly preferring dark mode interfaces and avoiding small typography. They are currently feeling the fatigue of a demanding debugging session but remain focused on their technical environment. Their communication style is direct, reflecting a professional yet weary emotional tone after a long day of coding.
