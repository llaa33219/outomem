# Quickstart Guide

Get up and running with outomem in minutes. This guide covers installation, setup, and basic memory operations.

## Prerequisites

Before you start, ensure you have the following ready:

- Python 3.10 or higher
- A running Neo4j instance (version 5.0+)
- OpenAI API key (or another supported provider)

## Install

Install the package using pip.

```bash
pip install outomem
```

Verify the installation by checking the version.

```bash
python -c "import outomem; print('outomem installed successfully')"
```

Expected output:
```text
outomem installed successfully
```

## Configure Neo4j

Outomem uses Neo4j to store relational facts. The easiest way to start is with Docker.

```bash
docker run \
    --name neo4j-outomem \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:5.12.0
```

Wait a few seconds for the database to initialize. You can access the browser interface at `http://localhost:7474`.

## Initialize Outomem

Create a new Python file and initialize the `Outomem` class. You must provide all configuration parameters including `embed_dim` to match your embedding model.

```python
from outomem import Outomem

# Initialize the memory system
outomem = Outomem(
    provider="openai",
    base_url="https://api.openai.com/v1",
    api_key="your-api-key",
    model="gpt-4",
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

print("Outomem initialized.")
```

**Important:** The `embed_dim` parameter must match your embedding model's output dimensions:

| Model | `embed_dim` |
| :--- | :--- |
| `all-MiniLM-L6-v2` | 384 |
| `text-embedding-3-small` | 768 |
| `text-embedding-3-large` | 3072 |

Expected output:
```text
Outomem initialized.
```

## Store Your First Memory

Use the `remember()` method to process a conversation. Outomem extracts personal preferences, facts, and temporal events automatically.

```python
conversation = [
    {"role": "user", "content": "Hi, I'm Luke. I really love drinking dark roast coffee in the morning."},
    {"role": "assistant", "content": "Nice to meet you, Luke! I'll remember that you enjoy dark roast coffee."}
]

outomem.remember(conversation)
print("Memory stored.")
```

Expected output:
```text
Memory stored.
```

## Retrieve Context

When you need to generate a response, use `get_context()` to pull relevant memories into your prompt.

```python
context = outomem.get_context("What does Luke like to drink?")
print(f"Context: {context}")
```

Expected output:
```text
Context: Luke prefers dark roast coffee in the morning.
```

## See Contradiction Detection

Outomem detects when a user changes their mind. If you store a conflicting preference, it deactivates the old one and records the change.

```python
# Luke changes his mind
outomem.remember("Actually, I've started to hate dark roast coffee. It's too bitter. I prefer light roast now.")

# Check the updated context
new_context = outomem.get_context("What are Luke's coffee preferences?")
print(f"Updated Context: {new_context}")
```

Expected output:
```text
Updated Context: Luke prefers light roast coffee. He previously liked dark roast but now finds it too bitter.
```

## Next Steps

Now that you've mastered the basics, explore these resources:

- [Architecture Overview](../architecture.md) - Learn how the four layers work.
- [Design Philosophy](../philosophy.md) - Understand the Korean first design approach.
- [API Reference](../api-reference/outomem.md) - Detailed documentation for the main Outomem class.
- [Backup & Restore](backup-restore.md) - Preserve memories when changing embedding models.
