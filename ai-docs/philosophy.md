# Outomem Design Philosophy

This document outlines the foundational principles and architectural decisions behind the outomem library. It serves as a guide for developers and users to understand the core logic of the ultimate memory system library for AI agents.

## Design Principles

Outomem follows five core principles that define its behavior and structure.

### Emotion Aware
Sentiment isn't just metadata in this system. It's a structural component. Every personalization entry undergoes sentiment analysis to determine its polarity and strength. This emotional context dictates how memories are stored, retrieved, and merged.

### Contradiction Aware
The system actively monitors for changes in user preferences. It detects contradictions by identifying sentiment polarity flips. If a user previously loved coffee but now expresses a dislike for it, the system recognizes this shift and updates the memory state accordingly.

### Layered Memory
Memory isn't a monolithic block. Outomem uses distinct layers for different purposes. Raw facts preserve original data, personalization tracks user preferences, long term storage holds factual knowledge, and temporal sessions record events and changes over time.

### Hybrid Storage
The library combines two powerful storage technologies. LanceDB provides fast vector similarity search for semantic retrieval. Neo4j enables complex graph traversals to manage relationships and contradiction chains.

### Memory Strength
Preferences aren't static. They have an importance level that naturally decays over time. This ensures that older, unused preferences fade away while frequently reinforced memories remain strong.

## Why Hybrid Storage

Neither vector search nor graph databases are sufficient on their own for a sophisticated memory system. Vector search through LanceDB allows the agent to find semantically similar memories quickly. However, it struggles with complex relationships and history tracking. Neo4j fills this gap by allowing the system to traverse contradiction chains and query deep relationships between different memory nodes. By using both, outomem achieves both speed and relational depth.

## Why Emotion Matters

Emotion is the key to understanding human preference. In outomem, sentiment analysis enables the detection of contradictions. When a new preference has an opposite sentiment to an existing one, the system can trigger a state change. Furthermore, emotional intensity determines the consolidation boost. Memories with higher emotional weight receive a larger strength increase during the merging process, reflecting how humans remember significant events more vividly.

## Why Memory Decays

Outomem uses a biological analogy for memory management. Just as human memories fade if they aren't recalled, preferences in this system lose strength over time. The system applies a decay formula where strength equals the initial value multiplied by 0.95 raised to the power of days elapsed. This recalculation happens during context retrieval, ensuring the agent always works with the most relevant and current information.

## Core Invariants

These rules must never be violated to maintain system integrity.

- Every write operation must go to both LanceDB and Neo4j to keep the hybrid storage synchronized.
- Contradictions must deactivate old entries rather than deleting them. This preserves the history of preference changes.
- Sentiment detection is mandatory for all personalization entries. The system cannot store a preference without understanding its emotional polarity.
