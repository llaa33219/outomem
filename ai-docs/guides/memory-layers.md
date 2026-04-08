# Memory Layers in Outomem

Outomem organizes information into four distinct layers to manage how an agent perceives and recalls data. Personal preferences, factual knowledge, and historical events remain organized and accessible through this structure.

## Layer Overview

| Layer | Purpose | Written | Read |
| :--- | :--- | :--- | :--- |
| raw_facts | Original conversation data | During extraction | Context synthesis |
| personalization | User preferences and sentiment | During extraction | Context synthesis |
| long_term | Factual knowledge | During extraction | Context synthesis |
| temporal_sessions | Events and preference changes | During extraction | Context synthesis |

## raw_facts

The `raw_facts` layer acts as the foundation of the memory system. Original content extracted from conversations stays here along with the full conversation text for context.

**Schema Fields:**
- `id`: Unique identifier.
- `content`: The extracted fact text.
- `conversation`: The full conversation snippet.
- `layer`: Always set to "raw_facts".
- `created_at`: Timestamp of creation.
- `vector`: Semantic embedding.

**Example Data:**
```json
{
  "content": "User likes dark mode",
  "conversation": "User: I really prefer dark mode for all my apps.\nAssistant: Noted!"
}
```

**Query Timing:**
This layer is queried during `get_context` to provide raw evidence for the LLM to synthesize.

## personalization

This layer tracks user-specific traits, likes, and dislikes. Sentiment analysis and a decay system help prioritize fresh information.

**Schema Fields:**
- `sentiment`: "positive", "negative", or "neutral".
- `strength`: Current importance (0.0 to 1.0).
- `is_active`: Boolean flag for validity.
- `contradiction_with`: ID of the fact this one replaces.

**Example Data:**
```json
{
  "content": "Likes spicy food",
  "sentiment": "positive",
  "strength": 1.0,
  "is_active": true
}
```

**Query Timing:**
Queried during every `get_context` call to ensure the agent respects user preferences.

## long_term

Consolidated factual knowledge lives here. Unlike personalization, these facts are generally objective or non-preference based.

**Schema Fields:**
- `source_facts`: List of raw fact IDs that contributed to this entry.
- `access_count`: Number of times this fact was retrieved.

**Deduplication:**
The system checks for existing facts using a 0.85 similarity threshold. If a match exists, it doesn't add a duplicate.

**Query Timing:**
Used to provide the agent with a stable knowledge base about the world or the user's environment.

## temporal_sessions

This layer records the timeline of events. Capturing when things happened and how preferences evolved is its main job.

**Schema Fields:**
- `session_id`: Groups events from the same interaction.
- `event_type`: "event" or "preference_change".
- `old_content` / `new_content`: Used for tracking preference flips.

**Example Data:**
```json
{
  "event_type": "preference_change",
  "old_content": "Likes coffee",
  "new_content": "Prefers tea now"
}
```

**Query Timing:**
Queried to give the agent a sense of time and recent history.

## How Categorization Works

The `remember()` method sends conversation text to an LLM with an extraction prompt. Classification into `personal`, `factual`, and `temporal` categories happens during this step.

After extraction, the system runs `_detect_sentiment` on personal facts. Keywords like "좋아" (like) or "싫어" (hate) help assign a sentiment. Identifying if the user is expressing a preference becomes easier with this data.

## Layer Interaction

Layers interact most dynamically when a user changes their mind. If a new personal fact has a different sentiment than an existing one and similarity is high (above 0.95), a contradiction occurs.

The system then deactivates the old personalization entry. Linking the new entry via the `contradiction_with` field happens next. A `preference_change` event is created in `temporal_sessions`.

This process ensures the agent doesn't get confused by outdated preferences while still remembering that a change happened.
