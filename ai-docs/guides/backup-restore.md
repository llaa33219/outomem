# Backup & Restore Guide

Outomem provides backup and restore functionality to preserve memory data when changing embedding models or migrating between environments.

## Why Backup?

When you change embedding models, the vector dimensions change (e.g., 384 → 768). Existing vectors become incompatible with the new schema. Backup allows you to:

1. **Preserve original text content** - All memories, preferences, and events.
2. **Re-embed with new model** - Generate new vectors matching the new dimensions.
3. **Maintain relationships** - Neo4j contradiction chains and event links remain intact.

## Creating a Backup

Export all memory data to a JSON file. Vectors are excluded; only text content and metadata are preserved.

```python
from outomem import Outomem

memory = Outomem(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    embed_api_url="https://api.openai.com/v1/embeddings",
    embed_api_key="sk-...",
    embed_model="text-embedding-3-small",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    embed_dim=384,  # Current model dimensions
)

# Export backup
memory.export_backup("./backup_20260412.json")
print("Backup created successfully.")
```

## Backup Format

The backup file is a JSON document with the following structure:

```json
{
    "version": "1.0",
    "exported_at": "2026-04-12T10:30:00+00:00",
    "embed_config": {"dimensions": 384},
    "lancedb": {
        "raw_facts": [
            {
                "id": "fact-123",
                "content": "User prefers dark roast coffee",
                "conversation": "user: I love dark roast...",
                "layer": "raw_facts",
                "created_at": "2026-01-15T08:30:00+00:00"
            }
        ],
        "long_term": [...],
        "personalization": [
            {
                "id": "pref-456",
                "content": "likes dark roast coffee",
                "category": "preference",
                "sentiment": "positive",
                "strength": 0.85,
                "is_active": true,
                "access_count": 12
            }
        ],
        "temporal_sessions": [...]
    },
    "neo4j": {
        "personalizations": [
            {
                "id": "pref-456",
                "content": "likes dark roast coffee",
                "relationships": [
                    {
                        "type": "CONTRADICTED_BY",
                        "target_id": "pref-789",
                        "timestamp": "2026-03-01T14:00:00+00:00"
                    }
                ]
            }
        ],
        "temporal_sessions": [...],
        "sessions": [...]
    }
}
```

### What's Included

| Layer | Fields |
| :--- | :--- |
| LanceDB | All fields except `vector` |
| Neo4j | All node properties except `vector`, plus relationships |

### What's Excluded

- **Vectors** - Re-embedded during restore
- **Raw `vector` arrays** - Not portable between models

## Restoring a Backup

Import the backup into a new Outomem instance with updated dimensions.

```python
from outomem import Outomem

# New instance with different embedding model
memory = Outomem(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o",
    embed_api_url="https://api.openai.com/v1/embeddings",
    embed_api_key="sk-...",
    embed_model="text-embedding-3-large",  # New model
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    embed_dim=3072,  # New dimensions
)

# Import with re-embedding
memory.import_backup("./backup_20260412.json", reembed=True)
print("Memories restored with new embeddings.")
```

## Migration Workflow

Complete workflow for changing embedding models:

```python
from outomem import Outomem

# Step 1: Export from old instance
old_memory = Outomem(
    ...,
    embed_model="text-embedding-3-small",
    embed_dim=768,
)
old_memory.export_backup("./migration_backup.json")

# Step 2: Import to new instance
new_memory = Outomem(
    ...,
    embed_model="text-embedding-3-large",
    embed_dim=3072,
)
new_memory.import_backup("./migration_backup.json", reembed=True)

# Step 3: Verify
status = new_memory.health_check()
print(f"Health: {status['healthy']}")
print(f"Personalizations: {status['lancedb']['stats']['personalization']}")
```

## Re-embed vs. No Re-embed

The `reembed` parameter controls behavior:

### `reembed=True` (Default)

- Re-embeds all content using current embedding function.
- Allows dimension changes (384 → 768 → 3072).
- Slower (calls embedding API for each content).

### `reembed=False`

- Validates that backup dimensions match current `embed_dim`.
- Raises `ValueError` if dimensions differ.
- Use case: Same model, migrating database only.

```python
# Fast restore (same model)
memory.import_backup("./backup.json", reembed=False)
```

## Error Handling

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `ValueError: Unsupported backup version` | Backup file format incompatible | Re-export from source instance |
| `ValueError: Cannot import without re-embedding` | `reembed=False` with dimension mismatch | Use `reembed=True` or match dimensions |
| `ValueError: Backup file must contain a JSON object` | Invalid JSON structure | Check backup file integrity |

## Best Practices

1. **Backup before model changes** - Always export before switching embedding models.
2. **Version your backups** - Include date in filename: `backup_20260412.json`.
3. **Verify after restore** - Run `health_check()` and test retrieval.
4. **Test with sample** - Import a small test backup first to validate dimensions.
5. **Keep backups secure** - Backup contains all user preferences and memories.

## Example: Disaster Recovery

```python
import os
from outomem import Outomem

# Daily backup routine
memory = Outomem(...)
backup_path = f"./backups/backup_{datetime.now().strftime('%Y%m%d')}.json"
memory.export_backup(backup_path)

# Recovery from backup
if os.path.exists("./backups/backup_20260412.json"):
    memory.import_backup("./backups/backup_20260412.json")
    print("Recovered from backup.")
```
