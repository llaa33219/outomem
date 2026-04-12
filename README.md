# outomem

<img src="logo.svg" alt="outomem logo" width="400">

Outomem is the ultimate memory system library for AI agents. This tool manages user preferences, finds contradictions, and builds context for agents. The system organizes data into four layers: personalization, long term, temporal sessions, and raw facts. It tracks sentiment and detects when a user changes their mind by looking for polarity flips. Memory strength decays over time to keep context fresh. We built it with a Korean first design approach.

## Installation

```bash
pip install outomem
```

Note: You need external database instances running.

## Quick Start

```python
from outomem import Outomem

# Initialize the memory system
memory = Outomem(
    provider="openai-responses",
    base_url="https://api.openai.com/v1/responses",
    api_key="your-api-key",
    model="gpt-5.4",
    embed_api_url="https://api.openai.com/v1/embeddings",
    embed_api_key="your-api-key",
    embed_model="text-embedding-3-small",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    db_path="./outomem.lance",
    style_path="./style.md",
    embed_dim=768,  # Match your embedding model dimensions
)

# Store a new memory
memory.remember("I prefer dark mode for all my applications.")

# Get context for a query
context = memory.get_context("What are the user's UI preferences?")
print(context)

# Backup before changing embedding model
memory.export_backup("./backup.json")
```

## Health Check

Verify that all memory system components are operational before processing requests.

```python
status = memory.health_check()

if status["healthy"]:
    print("All systems operational")
else:
    print(f"LanceDB: {status['lancedb']['connected']}")
    print(f"Neo4j: {status['neo4j']['connected']}")
    print(f"Embedding: {status['embedding']['working']}")
```

The `health_check()` method returns a dict with connection status for LanceDB, Neo4j, and the embedding function, plus table statistics and node counts.

## Philosophy

See [Design Philosophy](ai-docs/philosophy.md).

## Architecture Overview

Outomem uses a layered approach to manage different types of information. The vector store handles semantic retrieval while the graph store manages complex relationships between facts. This hybrid setup allows for both fast similarity search and deep graph traversal.

See [Architecture](ai-docs/architecture.md) for more details.

## API Overview

| Class | Description |
| :--- | :--- |
| Outomem | Main API for managing agent memory and context. |
| LayerManager | Handles vector storage and retrieval. |
| GraphLayerManager | Manages graph database operations for relational facts. |

## Documentation Index

### Core Concepts
- [Design Philosophy](ai-docs/philosophy.md)
- [Architecture](ai-docs/architecture.md)

### Guides
- [Quickstart](ai-docs/guides/quickstart.md)
- [Backup & Restore](ai-docs/guides/backup-restore.md)

### API Reference
- [Outomem](ai-docs/api-reference/outomem.md)
- [LayerManager](ai-docs/api-reference/layers.md)
- [Neo4jLayerManager](ai-docs/api-reference/neo4j-layers.md)

### Project Management
- [Governance](ai-docs/GOVERNANCE.md)

## Requirements

- Python >= 3.10
- External database instances (configurable)

## License

Apache License 2.0 - See [LICENSE](LICENSE)
