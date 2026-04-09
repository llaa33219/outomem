# System Architecture

This document describes the internal structure and data flow of the outomem AI agent memory system. It explains how the system manages user preferences, factual knowledge, and temporal events using a hybrid storage approach.

## Component Overview

Outomem coordinates several specialized components to provide a unified memory interface.

```text
       +----------+
       |   User   |
       +----+-----+
            |
            v
      +-----+-----+      +-------------------+
      |  Outomem  +------>   LLM Provider    |
      +-----+-----+      +-------------------+
            |
            +-----------------------+
            |                       |
            v                       v
    +-------+-------+       +-------+-------+
    | Layer Manager |       | Neo4j Manager |
    +-------+-------+       +-------+-------+
            |                       |
            v                       v
      +-----+-----+           +-----+-----+
      |  LanceDB  |           |   Neo4j   |
      +-----------+           +-----------+
```

### Core Components

- **[Outomem](api-reference/outomem.md)**: The primary entry point that orchestrates the memory pipeline.
- **[LayerManager](api-reference/layers.md)**: Manages vector storage and similarity search via LanceDB.
- **[Neo4jLayerManager](api-reference/neo4j-layers.md)**: Handles graph relationships, contradiction chains, and session history.
- **[LLMProvider](api-reference/providers.md)**: Interfaces with external language models for extraction and synthesis.

For the reasoning behind this hybrid approach, see [philosophy.md](philosophy.md).

## Data Flow: remember()

The `remember()` method processes new information and updates the memory stores.

```text
Conversation -> [Format] -> [LLM Extraction with Context Awareness]
                                                      |
      +----------------------------------------------+
      |
      v
[Contradiction Check] -> [Dual-Write] -> [Consolidation Trigger]
```

1. **Format conversation**: Converts input into a standardized list of message objects.
2. **LLM extraction with existing memory context**: The LLM receives existing memories as context to avoid duplicates and contradictions. It decides what is truly worth storing, assigns emotional intensity, and categorizes into personal/factual/temporal.
3. **Sentiment detection**: Assigns emotional polarity using keyword matching.
4. **Contradiction check**: Compares new personalizations against existing ones using vector similarity and sentiment polarity.
5. **Dual-write**: Persists data to both LanceDB and Neo4j simultaneously.
6. **Consolidation trigger**: If the system detects potential duplicates, it uses an LLM to merge similar memories.

## Data Flow: get_context()

The `get_context()` method retrieves and synthesizes relevant memories for a given query.

```text
Query -> [LLM Retrieval Planning] -> [Layer-Specific Search] -> [LLM Filter]
                                                               |
      +--------------------------------------------------------+
      |
      v
[Recalculate Strength] -> [Record Access] -> [LLM Synthesis] -> Context
```

1. **Format conversation**: Prepares the recent history for context retrieval.
2. **LLM Retrieval Planning**: The LLM analyzes the query and decides which layers to search and what search queries to use for each layer.
3. **Layer-specific search**: Each selected layer is searched with its own optimized embedding query.
4. **LLM Filter**: The LLM judges which retrieved memories are actually relevant to the query.
5. **Recalculate strengths**: Applies time-based decay to personalization entries.
6. **Record access**: Updates access counts and timestamps in both stores.
7. **LLM synthesis**: Combines retrieved facts into a coherent context string using the synthesis prompt.
8. **Token truncation**: Ensures the final output fits within the model's token limits.

## Storage Architecture

Outomem uses two distinct storage engines to balance speed and relational depth.

### LanceDB (Vector Store)

LanceDB stores high-dimensional vectors for semantic search. It maintains four tables:

- **raw_facts**: Original data points with their source conversation context.
- **long_term**: Consolidated factual knowledge.
- **personalization**: User preferences with strength and sentiment metadata.
- **temporal_sessions**: Log of events and state changes.

### Neo4j (Graph Store)

Neo4j manages the complex web of relationships between memory nodes.

- **Nodes**:
  - `Personalization`: Represents a specific user preference.
  - `TemporalSession`: Represents an event or a change in state.
  - `Session`: Groups temporal events into logical conversation blocks.
- **Relationships**:
  - `CONTRADICTED_BY`: Links an old preference to the new one that superseded it.
  - `HAS_EVENT`: Connects a Session to its constituent TemporalSessions.
  - `AFFECTED`: Links a TemporalSession to the Personalization it modified.

## Dual-Write Consistency

The system maintains consistency by writing to both LanceDB and Neo4j within the same `remember()` call. It doesn't use distributed transactions. Instead, it relies on eventual consistency. If one store fails, the system logs the error but continues operation to maintain availability.

## LLM Integration Points

Language models perform five critical tasks in the pipeline:

1. **Extraction**: Identifying structured facts from raw conversation text, with awareness of existing memories to avoid duplicates.
2. **Retrieval Planning**: Analyzing the query to decide which memory layers to search and generating optimized search queries per layer.
3. **Retrieval Filtering**: Judging which retrieved memories are actually relevant to the current query.
4. **Consolidation**: Merging overlapping or redundant personalization entries.
5. **Context Synthesis**: Turning a list of retrieved facts into a natural language prompt.

## Embedding Pipeline

The system uses a consistent embedding process for all vector operations:

- **Vector Dimension**: 384 (optimized for `all-MiniLM-L6-v2`).
- **Generation**: API-based embeddings computed during both write and read operations.
- **Similarity**: Cosine similarity is used for all retrieval tasks.
