# LayerManager API Reference

The `LayerManager` class handles vector storage and retrieval using LanceDB. It manages four distinct memory layers: `raw_facts`, `long_term`, `personalization`, and `temporal_sessions`.

## Constants

| Constant | Value | Description |
| :--- | :--- | :--- |
| `DEFAULT_VECTOR_DIM` | 384 | Default embedding dimensions for `all-MiniLM-L6-v2`. |
| `DEFAULT_LOCAL_MODEL` | "sentence-transformers/all-MiniLM-L6-v2" | Default local embedding model. |

## Schemas

### `build_schemas(vector_dim: int) -> dict[str, pa.Schema]`

Dynamically builds PyArrow schemas for all four memory tables with the specified vector dimensions.

**Parameters:**
- `vector_dim`: Embedding vector dimensions (e.g., 384, 768, 3072).

**Returns:** Dictionary mapping table names to PyArrow schemas.

**Source:** [layers.py:33-91](../../outomem/layers.py#L33)

## Constructor

### `LayerManager(db_path: str, embed_fn: EmbeddingFunction | None, vector_dim: int = 384)`

Initializes the LanceDB connection and ensures all layer tables exist with the correct vector dimensions.

- **db_path**: Path to the LanceDB database file.
- **embed_fn**: Optional custom embedding function. Defaults to `fastembed` with `all-MiniLM-L6-v2`.
- **vector_dim**: Embedding vector dimensions. Must match the output of `embed_fn`. Default: 384.

**Important:** The `vector_dim` parameter determines the schema for all tables. Changing it on an existing database requires re-embedding via `export_data()` and `import_data()`.

**Source:** [lines 95-105](../../outomem/layers.py#L95)

## EmbeddingFunction Protocol

A protocol defining the interface for embedding functions used by `LayerManager`.

### `__call__(texts: list[str]) -> list[list[float]]`
- **texts**: A list of strings to embed.
- **Returns**: A list of vector embeddings (list of floats).

**Source:** [lines 18-21](../../outomem/layers.py#L18)

## Layer Schemas

### `raw_facts`
Stores atomic facts extracted from conversations.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | utf8 | Unique identifier. |
| `content` | utf8 | The fact text. |
| `conversation` | utf8 | ID of the source conversation. |
| `layer` | utf8 | Always "raw_facts". |
| `created_at` | timestamp | Creation time (UTC). |
| `vector` | list[float32] | Embedding vector (dimensions set at initialization). |

**Source:** [lines 33-42](../../outomem/layers.py#L33)

### `long_term`
Stores consolidated, high-level knowledge.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | utf8 | Unique identifier. |
| `content` | utf8 | The consolidated knowledge. |
| `source_facts` | list[utf8] | IDs of source raw facts. |
| `layer` | utf8 | Always "long_term". |
| `created_at` | timestamp | Creation time (UTC). |
| `updated_at` | timestamp | Last update time (UTC). |
| `access_count` | int64 | Number of times retrieved. |
| `vector` | list[float32] | Embedding vector (dimensions set at initialization). |

**Source:** [lines 43-54](../../outomem/layers.py#L43)

### `personalization`
Stores user preferences, traits, and habits with decay mechanics.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | utf8 | Unique identifier. |
| `content` | utf8 | The preference or trait. |
| `category` | utf8 | Category (e.g., "preference"). |
| `sentiment` | utf8 | "positive", "negative", or "neutral". |
| `layer` | utf8 | Always "personalization". |
| `created_at` | timestamp | Creation time (UTC). |
| `updated_at` | timestamp | Last update time (UTC). |
| `strength` | float64 | Current memory strength (0.0 to 1.0). |
| `decay_factor` | float64 | Rate of decay over time. |
| `initial_strength` | float64 | Strength at creation. |
| `last_accessed` | timestamp | Last time the memory was used. |
| `access_count` | int64 | Number of times retrieved. |
| `contradiction_with` | utf8 | ID of contradictory fact. |
| `is_active` | bool | False if superseded. |
| `vector` | list[float32] | Embedding vector (dimensions set at initialization). |

**Source:** [lines 55-75](../../outomem/layers.py#L55)

### `temporal_sessions`
Tracks session-specific events and state changes.

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | utf8 | Unique identifier. |
| `session_id` | utf8 | ID of the session. |
| `event_type` | utf8 | Type of event. |
| `content` | utf8 | Event description. |
| `timestamp` | timestamp | Event time (UTC). |
| `layer` | utf8 | Always "temporal_sessions". |
| `metadata` | string | JSON metadata. |
| `vector` | list[float32] | Embedding vector (dimensions set at initialization). |
| `related_personalization_id` | utf8 | ID of linked personalization. |
| `old_content` | utf8 | Previous state content. |
| `new_content` | utf8 | New state content. |

**Source:** [lines 76-90](../../outomem/layers.py#L76)

## CRUD Methods

### Raw Facts
| Method | Description | Line |
| :--- | :--- | :--- |
| `add_raw_fact(content, conversation)` | Adds a new raw fact. | 130 |

### Long Term
| Method | Description | Line |
| :--- | :--- | :--- |
| `add_long_term(content, source_facts)` | Adds consolidated knowledge. | 146 |
| `update_long_term(id, content)` | Updates existing long-term memory. | 165 |
| `get_all_long_term()` | Returns all long-term memories. | 396 |

### Personalization
| Method | Description | Line |
| :--- | :--- | :--- |
| `add_personalization(...)` | Adds a new user preference. | 179 |
| `update_personalization_strength(id, delta)` | Adjusts strength by delta. | 214 |
| `record_access(id)` | Updates `last_accessed` timestamp. | 228 |
| `deactivate_personalization(id)` | Sets `is_active` to False. | 465 |
| `boost_personalization_strength(id, boost)` | Increases strength and updates access. | 472 |
| `get_all_personalizations()` | Returns all personalization records. | 383 |
| `merge_personalizations(ids, new_content, boost)` | Merges multiple records into one. | 317 |

### Temporal
| Method | Description | Line |
| :--- | :--- | :--- |
| `add_temporal(...)` | Records a session event. | 338 |
| `get_recent_temporal(limit)` | Returns recent events sorted by time. | 387 |

## Search Methods

| Method | Description | Line |
| :--- | :--- | :--- |
| `search(layer, query_embedding, limit)` | Vector search in a specific layer. | 370 |
| `find_similar_personalizations(content, threshold)` | Finds similar preferences. | 400 |
| `find_active_similar_personalizations(content, threshold)` | Finds active similar preferences. | 431 |
| `find_similar_long_term(content, threshold)` | Finds similar long-term knowledge. | 491 |

## Decay Methods

| Method | Description | Line |
| :--- | :--- | :--- |
| `recalculate_all_strengths()` | Applies time-based decay to all records. | 272 |
| `recalculate_and_apply_boost(id, repeat_count, base_strength)` | Decays then boosts a specific record. | 235 |
| `decay_personalization(factor)` | Multiplies all strengths by a factor. | 311 |

## Health Check Methods

| Method | Description | Line |
| :--- | :--- | :--- |
| `check_connection()` | Tests if LanceDB connection is alive. | 521 |
| `check_tables()` | Verifies all four tables exist and are accessible. | 529 |
| `get_table_stats()` | Returns row counts for each table. | 543 |
| `check_embedding(test_text)` | Confirms embedding function returns correct dimension. | 553 |

### `check_connection() -> bool`
Returns `True` if `list_tables()` succeeds without exception.

### `check_tables() -> dict[str, bool]`
Returns a dict mapping each table name to its accessibility status.

### `get_table_stats() -> dict[str, int]`
Returns row counts. Returns `-1` for tables that fail to open.

### `check_embedding(test_text: str = "health check test") -> bool`
Validates that the embedding function returns a float list matching the configured `vector_dim`.

## Backup Methods

These methods enable data export and import for migration between embedding models.

### `export_data() -> dict[str, Any]`

Exports all four table contents without vectors. Used internally by `Outomem.export_backup()`.

**Source:** [layers.py:167](../../outomem/layers.py#L167)

**Returns:**
```python
{
    "raw_facts": [{"id": "...", "content": "...", "created_at": "ISO", ...}],
    "long_term": [...],
    "personalization": [...],
    "temporal_sessions": [...]
}
```

Vectors are excluded from the export. Timestamps are serialized to ISO format strings.

### `import_data(data: dict[str, Any], embed_fn: EmbeddingFunction) -> None`

Imports data from a backup dictionary, re-embedding all content. Used internally by `Outomem.import_backup()`.

**Source:** [layers.py:181](../../outomem/layers.py#L181)

**Parameters:**
- `data`: Backup dictionary with table contents.
- `embed_fn`: Embedding function to re-embed all content.

**Behavior:**
1. Clears all existing table data.
2. Re-embeds all `content` fields using `embed_fn`.
3. Restores all metadata (strength, access_count, timestamps, etc.).
4. Preserves original IDs for cross-DB relationship consistency.
