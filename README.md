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
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4o",
    embed_api_url="https://api.openai.com/v1",
    embed_api_key="your-api-key",
    embed_model="text-embedding-3-small",
    db_config={
        "vector_store": "lancedb",
        "vector_path": "./outomem.lance",
        "graph_store": "neo4j",
        "graph_uri": "bolt://localhost:7687",
        "graph_user": "neo4j",
        "graph_password": "password",
    },
    style_path="./style.md"
)

# Store a new memory
memory.remember("I prefer dark mode for all my applications.")

# Get context for a query
context = memory.get_context("What are the user's UI preferences?")
print(context)
```

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

### Project Management
- [Governance](ai-docs/GOVERNANCE.md)

## Requirements

- Python >= 3.10
- External database instances (configurable)

## License

MIT
