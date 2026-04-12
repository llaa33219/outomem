# Neo4jLayerManager API Reference

The `Neo4jLayerManager` class manages graph database operations for relational facts in the Outomem system. It handles complex relationships, contradiction chains, and temporal event tracking using Neo4j.

## Constructor

`Neo4jLayerManager(uri: str, auth: tuple[str, str], database: str = "neo4j")` [Line 15]

Initializes the Neo4j driver and sets up required constraints.

- `uri`: Neo4j connection URI (e.g., `bolt://localhost:7687`).
- `auth`: A tuple containing `(username, password)`.
- `database`: The name of the Neo4j database to use.

## Graph Schema

The system maintains a specific schema to track personalizations and their history.

### Node Types
- **Personalization**: Stores user preferences, categories, and strength scores.
- **TemporalSession**: Represents a specific event or interaction.
- **Session**: Groups multiple temporal events into a single session.

### Relationship Types
- `CONTRADICTED_BY`: Links an old personalization to a new one that replaces it.
- `HAS_EVENT`: Connects a `Session` to its `TemporalSession` events.
- `AFFECTED`: Links a `TemporalSession` to the `Personalization` it modified or created.

### Constraints
The following constraints are created on startup [Lines 31-35]:
- Unique ID for `Personalization` nodes.
- Unique ID for `TemporalSession` nodes.
- Unique ID for `Session` nodes.

## Personalization Methods

### `add_personalization()` [Line 55]
Adds a new personalization node. If `contradiction_with` is provided, it creates a `CONTRADICTED_BY` edge from the old node to the new one and deactivates the old node.

### `get_personalization(id: str)` [Line 113]
Retrieves a single personalization by its ID.

### `get_all_personalizations(active_only: bool = False)` [Line 123]
Returns all personalizations, optionally filtering for only active ones.

### `find_active_similar_personalizations()` [Line 134]
Finds active personalizations similar to the provided content using cosine similarity on stored vectors.

### `update_personalization_strength(id: str, delta: float)` [Line 198]
Adjusts the strength of a personalization by a specific amount, clamped between 0.0 and 1.0.

### `boost_personalization_strength(id: str, boost: float = 0.15)` [Line 214]
Increases the strength of a personalization and updates its last accessed timestamp.

### `record_access(id: str)` [Line 235]
Updates the `last_accessed` timestamp and increments the `access_count`.

### `deactivate_personalization(id: str)` [Line 246]
Sets `is_active` to false for a specific personalization.

### `merge_personalizations()` [Line 273]
Merges multiple personalizations into a single new one, deleting the old nodes.

### `recalculate_all_strengths()` [Line 256]
Applies time based decay to all active personalizations based on their `decay_factor` and time since last access.

## Temporal Methods

### `add_temporal()` [Line 321]
Records a temporal event. It merges the `Session` node and creates a `TemporalSession` node with `HAS_EVENT` and optional `AFFECTED` relationships.

### `get_recent_temporal(limit: int = 10)` [Line 373]
Retrieves the most recent temporal events across all sessions.

### `get_session_events(session_id: str)` [Line 386]
Returns all events associated with a specific session, ordered chronologically.

### `get_personalization_events(personalization_id: str)` [Line 398]
Returns all temporal events that affected a specific personalization.

## Graph Traversal Methods

Neo4j allows for deep traversal of relationships that are difficult to query in standard vector databases.

### `get_contradiction_chain(personalization_id: str)` [Line 412]
Traverses the `CONTRADICTED_BY` relationships to show the history of how a preference changed over time.

**Example Cypher:**
```cypher
MATCH path = (start:Personalization {id: $id})-[:CONTRADICTED_BY*]->(end)
RETURN nodes(path) AS chain
```

### `get_related_personalizations(personalization_id: str)` [Line 425]
Finds other personalizations that were affected by the same temporal events. This helps identify clusters of related preferences.

**Example Cypher:**
```cypher
MATCH (p:Personalization {id: $id})<-[:AFFECTED]-(t:TemporalSession)
MATCH (t)-[:AFFECTED]->(related:Personalization)
WHERE related.id <> $id
RETURN DISTINCT related
```

## Comparison with LayerManager

| Feature | LayerManager (LanceDB) | Neo4jLayerManager |
| :--- | :--- | :--- |
| Primary Storage | Vector / Local File | Graph / Remote Server |
| Search Type | Semantic Similarity | Relational / Traversal |
| Contradiction Tracking | Metadata only | Explicit `CONTRADICTED_BY` edges |
| Temporal History | Flat list | Session based graph |
| Strength Decay | Supported | Supported with Cypher |
| Multi-hop Queries | No | Yes (e.g., contradiction chains) |

## Key Cypher Patterns

### Contradiction and Deactivation
When a new personalization replaces an old one, the system deactivates the old node and links them [Line 91]:
```cypher
MATCH (old:Personalization {id: $old_id})
CREATE (old)-[:CONTRADICTED_BY]->(new:Personalization {...})
SET old.is_active = false
```

### Time Based Decay
The system uses Cypher's duration functions to calculate decay [Line 262]:
```cypher
WITH p, duration.between(p.last_accessed, datetime()).days AS days_elapsed
SET p.strength = p.initial_strength * p.decay_factor ^ days_elapsed
```

## Health Check Methods

### `check_connection() -> bool` [Line 440]
Executes `RETURN 1 AS health` to verify the driver can reach the database. Returns `False` on `Neo4jError` or `ServiceUnavailable`.

### `get_node_counts() -> dict[str, int]` [Line 448]
Returns counts for three node types:
- `personalization`: Total Personalization nodes.
- `temporal_session`: Total TemporalSession nodes.
- `session`: Total Session nodes.

Returns `-1` for any query that fails.

## Backup Methods

These methods enable data export and import for migration between embedding models.

### `export_data() -> dict[str, Any]`

Exports all nodes and relationships without vectors. Used internally by `Outomem.export_backup()`.

**Source:** [neo4j_layers.py:82](../../outomem/neo4j_layers.py#L82)

**Returns:**
```python
{
    "personalizations": [
        {
            "id": "...",
            "content": "...",
            "category": "...",
            "relationships": [
                {"type": "CONTRADICTED_BY", "target_id": "...", "timestamp": "..."}
            ]
        }
    ],
    "temporal_sessions": [
        {
            "id": "...",
            "content": "...",
            "relationships": [
                {"type": "AFFECTED", "target_id": "..."}
            ]
        }
    ],
    "sessions": [
        {
            "id": "...",
            "relationships": [
                {"type": "HAS_EVENT", "target_id": "..."}
            ]
        }
    ]
}
```

Vectors are excluded from personalization nodes. Relationships are serialized as arrays within each node.

### `import_data(data: dict[str, Any], embed_fn) -> None`

Imports data from a backup dictionary, re-embedding all content. Used internally by `Outomem.import_backup()`.

**Source:** [neo4j_layers.py:178](../../outomem/neo4j_layers.py#L178)

**Parameters:**
- `data`: Backup dictionary with node contents and relationships.
- `embed_fn`: Embedding function to re-embed all personalization content.

**Behavior:**
1. Clears all existing nodes (`DETACH DELETE`).
2. Re-embeds all personalization content using `embed_fn`.
3. Creates all nodes with original IDs and metadata.
4. Restores all relationships (`CONTRADICTED_BY`, `HAS_EVENT`, `AFFECTED`).
