# Contradiction Detection Guide

Outomem tracks user preferences over time. When a user changes their mind, the system detects the contradiction, deactivates the old preference, and records the change. This ensures the agent always works with the most current context while preserving history.

## What is a Contradiction

A contradiction occurs when a new personalization fact has an opposite sentiment polarity compared to an existing, semantically similar fact. The system uses a vector similarity threshold of 0.85 to identify related topics. If the sentiment flips from positive to negative, or vice versa, it triggers the contradiction flow.

## Detection Flow

The detection process happens within the `remember()` method in `core.py`.

1. **Sentiment Detection**: The system analyzes the new fact using keyword counting. It checks against `POSITIVE_KEYWORDS` and `NEGATIVE_KEYWORDS` (lines 109, 117).
2. **Similarity Search**: It searches LanceDB for active personalizations with a semantic similarity of 0.85 or higher (line 155).
3. **Sentiment Comparison**: The `_is_contradictory` helper checks if both facts have non-neutral sentiments and if those sentiments differ (line 119).
4. **Conflict Resolution**: If a contradiction exists, the system proceeds to update the memory layers.

## What Happens on Detection

When the system identifies a contradiction, it performs several atomic updates to maintain consistency.

### 1. Deactivation
The old fact is never deleted. Instead, its `is_active` flag is set to `false` in both LanceDB and Neo4j. This preserves the historical record for future analysis.

### 2. Temporal Recording
A `preference_change` event is recorded in the temporal layer. This event includes the old content, the new content, and a human readable summary like "취향 변화: [Old] → [New]".

### 3. New Fact Storage
The new fact is stored with a `contradiction_with` field containing the ID of the superseded fact.

### 4. Graph Relationship
In Neo4j, a `CONTRADICTED_BY` edge is created from the old node to the new node. This edge includes a timestamp of when the change occurred.

## Sentiment Detection Logic

The system uses a straightforward keyword based approach in `core.py`:

```python
def _detect_sentiment(self, text: str) -> str:
    text_lower = text.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"
```

## Practical Example

Consider a user who previously said, "I love coffee." Later, they say, "I hate coffee."

### Step 1: Initial State
- **LanceDB**: Active record with content "I love coffee", sentiment "positive", and ID `uuid-1`.
- **Neo4j**: Node `(p1:Personalization {content: "I love coffee", is_active: true})`.

### Step 2: Processing "I hate coffee"
1. `_detect_sentiment("I hate coffee")` returns "negative" (line 154).
2. `find_active_similar_personalizations` finds `uuid-1` with high similarity (line 155).
3. `_is_contradictory("positive", "negative")` returns `True` (line 160).

### Step 3: Final State
- **LanceDB**: 
    - `uuid-1` is now `is_active: false`.
    - New record `uuid-2` created with `contradiction_with: "uuid-1"`.
    - Temporal event recorded with `old_content: "I love coffee"` and `new_content: "I hate coffee"`.
- **Neo4j**:
    - `(p1)` updated to `is_active: false`.
    - `(p1)-[:CONTRADICTED_BY]->(p2)` relationship created.

## Querying Contradiction History

You can retrieve the full history of a user's changing preferences using the `get_contradiction_chain()` method in `Neo4jLayerManager`. This performs a variable length path traversal:

```cypher
MATCH path = (start:Personalization {id: $id})-[:CONTRADICTED_BY*]->(end)
RETURN nodes(path) AS chain
```

This returns the entire sequence of facts, allowing you to see how a user's tastes evolved over multiple sessions.
