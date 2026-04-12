# Outomem API Reference

The `Outomem` class serves as the primary interface for the memory system. It coordinates between LanceDB for vector storage and Neo4j for graph relationships.

## Import and Construction

Initialize the memory system with LLM provider details and database connection strings.

**Source:** `core.py:64-78`

### Constructor

```python
def __init__(
    self,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    embed_api_url: str,
    embed_api_key: str,
    embed_model: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    db_path: str = "./outomem.lance",
    style_path: str = "./style.md",
    embed_dim: int = 384,
) -> None
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `provider` | `str` | LLM provider name (e.g., "openai"). |
| `base_url` | `str` | Base URL for the LLM API. |
| `api_key` | `str` | API key for the LLM provider. |
| `model` | `str` | Model name for completions. |
| `embed_api_url` | `str` | API URL for generating embeddings. |
| `embed_api_key` | `str` | API key for the embedding service. |
| `embed_model` | `str` | Model name for embeddings. |
| `neo4j_uri` | `str` | Connection URI for Neo4j. |
| `neo4j_user` | `str` | Username for Neo4j authentication. |
| `neo4j_password` | `str` | Password for Neo4j authentication. |
| `db_path` | `str` | Path to the LanceDB database file. |
| `style_path` | `str` | Path to the markdown file defining the agent's style. |
| `embed_dim` | `int` | Embedding vector dimensions. Default: 384 (for `all-MiniLM-L6-v2`). Set to match your embedding model (e.g., 768 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`). |

**Important:** The `embed_dim` parameter determines the vector schema for all LanceDB tables. Changing the embedding model requires re-embedding via `export_backup()` and `import_backup()`. See [Backup & Restore](../guides/backup-restore.md).

## Core Methods

### `remember(conversation)`

Extracts and stores information from a conversation history.

**Source:** `core.py:130`

*   **Input:** `list[dict[str, str]] | str`
*   **Return:** `None`

**Behavioral Pipeline:**
1.  **Extraction:** Uses the LLM to identify personal, factual, and temporal data.
2.  **Categorization:** Sorts data into specific memory layers.
3.  **Sentiment:** Detects polarity in personal facts using keyword matching.
4.  **Contradiction:** Checks if new information flips the polarity of existing active memories.
5.  **Storage:** Commits data to both LanceDB and Neo4j.
6.  **Consolidation:** Merges similar or redundant personalizations to maintain a clean state.

**Example:**
```python
memory.remember([
    {"role": "user", "content": "I love spicy food but hate cilantro."},
    {"role": "assistant", "content": "Noted! I'll remember your preferences."}
])
```

### `get_context(full_history, max_tokens)`

Retrieves and synthesizes relevant memories into a context string for the LLM.

**Source:** `core.py:366`

*   **Input:** `full_history: list[dict[str, str]] | str | None`, `max_tokens: int = 4096`
*   **Return:** `str`

**Pipeline Description:**
1.  **Embedding:** Generates a vector for the current conversation state.
2.  **Retrieval:** Performs similarity searches across personalization, long-term, temporal, and raw fact layers.
3.  **Maintenance:** Recalculates memory strengths (applying decay) and records access timestamps.
4.  **Synthesis:** Passes retrieved facts to the LLM to generate a coherent context block.
5.  **Fallback:** Uses a local Korean template if the LLM synthesis fails.

**Example:**
```python
context = memory.get_context("What should we have for dinner?")
# Returns: "The user loves spicy food but dislikes cilantro..."
```

### `health_check()`

Checks if all memory system components are operational.

**Source:** `core.py:580`

*   **Input:** None
*   **Return:** `dict[str, Any]`

**Check Components:**
1.  **LanceDB:** Verifies database connection and all four tables exist.
2.  **Embedding:** Confirms the embedding function returns vectors of correct dimension (384).
3.  **Neo4j:** Tests graph database connectivity.

**Return Structure:**
```python
{
    "healthy": bool,           # True only if all checks pass
    "lancedb": {
        "connected": bool,     # LanceDB connection status
        "tables": {            # Per-table existence
            "raw_facts": bool,
            "long_term": bool,
            "personalization": bool,
            "temporal_sessions": bool,
        },
        "stats": {             # Row counts per table
            "raw_facts": int,
            "long_term": int,
            "personalization": int,
            "temporal_sessions": int,
        },
    },
    "embedding": {
        "working": bool,       # Embedding function status
    },
    "neo4j": {
        "connected": bool,     # Neo4j connection status
        "node_counts": {       # Node counts by type
            "personalization": int,
            "temporal_session": int,
            "session": int,
        },
    },
}
```

**Example:**
```python
status = memory.health_check()
if not status["healthy"]:
    print(f"LanceDB: {status['lancedb']['connected']}")
    print(f"Neo4j: {status['neo4j']['connected']}")
    print(f"Embedding: {status['embedding']['working']}")
```

### `export_backup(path: str)`

Exports all memory data to a JSON backup file. Vectors are excluded; only text content and metadata are preserved.

**Source:** `core.py:635`

*   **Input:** `path: str` - Output file path.
*   **Return:** `None`

**Backup Format:**
```json
{
    "version": "1.0",
    "exported_at": "2026-04-12T10:30:00+00:00",
    "embed_config": {"dimensions": 384},
    "lancedb": {
        "raw_facts": [{"id": "...", "content": "...", ...}],
        "long_term": [...],
        "personalization": [...],
        "temporal_sessions": [...]
    },
    "neo4j": {
        "personalizations": [{"id": "...", "content": "...", "relationships": [...]}],
        "temporal_sessions": [...],
        "sessions": [...]
    }
}
```

**Use Cases:**
- Migrating to a different embedding model (different dimensions).
- Creating disaster recovery backups.
- Transferring memory between environments.

**Example:**
```python
memory.export_backup("./backup_20260412.json")
```

### `import_backup(path: str, reembed: bool = True)`

Imports memory data from a JSON backup file. Re-embeds all content using the current embedding function.

**Source:** `core.py:646`

*   **Input:**
    *   `path: str` - Backup file path.
    *   `reembed: bool` - If `True`, re-embeds all content. If `False`, validates that backup dimensions match current `embed_dim` (raises `ValueError` on mismatch).
*   **Return:** `None`

**Behavior:**
1. Validates backup version (must be "1.0").
2. If `reembed=False`, checks that backup dimensions match current `embed_dim`.
3. Clears existing data in both LanceDB and Neo4j.
4. Re-embeds all content using the current embedding function.
5. Restores all data including relationships and metadata.

**Example:**
```python
# After changing embedding model
memory = Outomem(..., embed_dim=768)  # New model
memory.import_backup("./backup_20260412.json", reembed=True)
```

## Private Methods

These methods handle internal logic for sentiment and memory maintenance.

### `_detect_sentiment(text: str) -> str`
Performs keyword-based sentiment analysis. Returns "positive", "negative", or "neutral".
**Source:** `core.py:109`

### `_is_contradictory(sentiment1: str, sentiment2: str) -> bool`
Returns `True` if both sentiments are non-neutral and differ from each other.
**Source:** `core.py:119`

### `_recalculate_strengths() -> None`
Triggers strength decay and updates across all storage backends.
**Source:** `core.py:126`

## Constants

### `POSITIVE_KEYWORDS`
Keywords that trigger a positive sentiment score.
- 좋아, 사랑, 싫어, 최고, 잘, 재밌, 행복, 짱
- good, like, love, best, great, awesome, amazing, fantastic, wonderful, excellent, prefer

### `NEGATIVE_KEYWORDS`
Keywords that trigger a negative sentiment score.
- 싫어, 可恶
- bad, hate, dislike, worst, terrible, awful, horrible, disgusting, annoying

## Error Handling

- **LLM Failures:** If the context synthesis fails, the system uses `_fallback_context` to generate a summary in Korean.
- **JSON Parsing:** Uses `safe_json_parse` to handle malformed LLM responses without crashing.

## Example Usage

```python
from outomem import Outomem

memory = Outomem(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    embed_api_url="https://api.openai.com/v1",
    embed_api_key="sk-...",
    embed_model="text-embedding-3-small",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    embed_dim=768,  # Match your embedding model dimensions
)

# Store information
memory.remember("My name is Luke and I'm a software engineer.")

# Retrieve context
context = memory.get_context("Who am I?")
print(context)

# Backup before changing embedding model
memory.export_backup("./backup.json")
```
