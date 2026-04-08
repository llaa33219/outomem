# Memory Consolidation

Memory consolidation is the process of merging similar or duplicate memories into a single, more powerful entry. Instead of keeping multiple records of the same preference, the system combines them to reinforce the user's identity and reduce noise.

## When It Triggers

Consolidation happens during the `remember()` cycle. If the system detects that a new memory is highly similar to existing ones, it sets `duplicates_found=True`. This flag triggers a background process that pulls all current personalization layers and sends them to an LLM for deduplication.

## The Consolidation Prompt

The system uses a specific prompt to guide the LLM in identifying what to merge. It looks like this:

```text
Analyze these user preference memories. Find duplicates and determine emotional intensity for reinforcement.

MEMORIES:
{memories}

RULES:
- Same topic/sentiment, different words -> MERGE (not duplicate entries)
- Merge uses MOST EXPRESSIVE wording
- After merge, assign a boost value based on emotional intensity of the merged facts:
  * "미치게", "흥분", "죽겠", "쾌감", "영원히", "최고", "heaven", "oxygen", "life meaning", "불타" -> boost: 0.25-0.35 (강함)
  * "좋아", "재밌", "행복", "짱", "love", "crazy", "motivation" -> boost: 0.15-0.20 (보통)
  * 감정이 약하면 -> boost: 0.05-0.10 (약함)

Output JSON:
{
  "consolidated": [
    {"content": "most expressive wording here", "original_ids": ["id1", "id2"], "boost": 0.28}
  ],
  "unique": ["id3"]
}
```

## Emotional Boost System

The system calculates a "boost" value based on the language used in the memories. Stronger emotions lead to a more significant increase in memory strength.

| Tier | Keywords | Boost Range |
| :--- | :--- | :--- |
| Strong (강함) | "미치게", "흥분", "죽겠", "쾌감", "영원히", "최고", "heaven", "oxygen", "life meaning", "불타" | 0.25, 0.35 |
| Normal (보통) | "좋아", "재밌", "행복", "짱", "love", "crazy", "motivation" | 0.15, 0.20 |
| Weak (약함) | Any other words | 0.05, 0.10 |

## The Merge Operation

When the LLM identifies memories to merge, the system performs these steps:

1.  **Calculate Strength**: It finds the maximum strength among the original memories.
2.  **Apply Boost**: It adds the LLM-provided boost to that maximum strength, capping the result at 1.0.
3.  **Delete Originals**: The system removes the old, redundant entries from storage.
4.  **Create New**: It inserts a new memory with the combined content and updated strength.

## Dual-Store Merge

Outomem maintains consistency across its hybrid storage. Every merge operation updates both LanceDB and Neo4j simultaneously. LanceDB handles the vector search updates, while Neo4j updates the graph relationships and properties. This ensures that semantic searches and relationship traversals always return the most current, consolidated information.

## Example

Imagine a user says two things over time:
1.  "I love dark roast." (Strength: 0.6)
2.  "I enjoy strong coffee." (Strength: 0.5)

The system identifies these as similar. The LLM chooses the most expressive wording, perhaps "I love strong dark roast coffee." Since "love" is a "Normal" tier keyword, it might assign a boost of 0.18.

The final merged memory becomes:
- **Content**: "I love strong dark roast coffee."
- **Strength**: 0.78 (0.6 max + 0.18 boost)

The original two entries are deleted, leaving one stronger preference in the system.
