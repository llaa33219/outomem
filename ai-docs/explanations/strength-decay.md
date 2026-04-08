# Memory Strength and Decay

Outomem uses a time-based decay system to manage how much influence a memory has on the agent's context. This ensures that the agent stays focused on what matters now, rather than getting bogged down by outdated preferences or one-off comments from months ago.

## Why Decay

Biological memory isn't a static database. It's a dynamic system where recent or frequently accessed information stays sharp, while old, unused details fade away. Outomem mimics this behavior. If you tell the agent you love coffee today, that's a strong signal. If you don't mention it again for six months, that preference weakens. This allows the agent to adapt as your tastes change without needing manual cleanup.

## The Formula

The system calculates memory strength using an exponential decay formula. Every day that passes without the memory being accessed reduces its strength.

`strength = initial_strength * decay_factor ^ days_elapsed`

*   **initial_strength**: The strength value when the memory was created or last reinforced.
*   **decay_factor**: Set to `0.95`. This means a memory loses 5% of its remaining strength every day.
*   **days_elapsed**: The time since the memory was last accessed or updated.

## Decay Curve Table

This table shows how a memory with an initial strength of 1.0 fades over time if it isn't reinforced.

| Time Elapsed | Strength | Status |
| :--- | :--- | :--- |
| Day 0 | 1.00 | Fresh |
| Day 1 | 0.95 | Strong |
| Day 7 | ~0.70 | Relevant |
| Day 30 | ~0.21 | Fading |
| Day 90 | ~0.01 | Ghost |
| Day 180 | ~0.00 | Negligible |
| Day 365 | ~0.00 | Forgotten |

## Bounds

Memory strength always stays within the range of `[0.0, 1.0]`. The system clamps values to ensure they never drop below zero or exceed one. Even with massive reinforcement, a memory can't become "super-strong" beyond the 1.0 limit.

## Initial Strength

Not all memories start equal. The system assigns different starting points based on the nature of the information.

*   **New Preferences**: Standard new memories start at `1.0`.
*   **Contradictions**: When you change your mind (e.g., "I used to like X, but now I like Y"), the new preference starts at `0.8`. This gives the new preference a slight "probationary" period while the old one is deactivated.

## Reinforcement

Memories don't just fade. They can be strengthened through use.

*   **Access Reset**: Every time a memory is retrieved for context, its `last_accessed` timestamp resets. This pauses the decay.
*   **Manual Boost**: The `boost_personalization_strength()` method adds a specific value to the current strength.
*   **Consolidation**: When similar memories merge, the new consolidated memory gets a boost based on emotional intensity.

## When Recalculation Happens

Recalculating strengths for every memory on every single access would be slow. Instead, Outomem only triggers the decay logic during `get_context()`. This happens just before the system synthesizes the final context for the agent. It ensures the agent sees the most accurate, up-to-date strengths without sacrificing performance during simple storage operations.

## Consolidation Boost

During memory consolidation, the LLM analyzes the emotional weight of the facts being merged. It assigns a boost value that reflects how much the user seems to care about the topic.

*   **Strong (0.25 to 0.35)**: High intensity words like "최고", "미치게", "영원히", or "life meaning".
*   **Normal (0.15 to 0.20)**: Positive but standard words like "좋아", "행복", "love", or "motivation".
*   **Weak (0.05 to 0.10)**: Low intensity or purely factual statements with little emotional coloring.
