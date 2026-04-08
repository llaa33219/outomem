# Sentiment Tracking

Outomem tracks the emotional tone of user statements to build a more accurate memory of preferences. This process helps the system distinguish between positive and negative associations with the same topic.

## Why Track Sentiment

Tracking sentiment enables the system to detect contradictions. Without this layer, a user saying "I love coffee" and later saying "I hate coffee" would simply look like two facts about coffee. By assigning a sentiment score, Outomem can identify when a user has changed their mind or expressed a conflicting preference. This aligns with our [emotion-aware principle](../philosophy.md), ensuring the agent understands the weight and polarity of user statements.

## The Keyword Approach

Outomem uses a deterministic keyword matching system rather than relying on an LLM for every sentiment check. This choice involves a specific trade-off.

Keyword matching is extremely fast and produces predictable results. It doesn't require expensive API calls or introduce the latency of a large language model. However, it lacks the nuance of modern NLP. It cannot easily handle complex sentence structures or subtle emotional shifts that an LLM might catch. We prioritize speed and reliability for this core memory function.

## Keyword Lists

The system uses predefined sets of positive and negative terms across multiple languages.

### Positive Keywords
These terms signal a favorable preference:
*   **Korean:** 좋아, 사랑, 싫어 (Note: currently in both lists), 최고, 잘, 재밌, 행복, 짱
*   **English:** good, like, love, best, great, awesome, amazing, fantastic, wonderful, excellent, prefer

### Negative Keywords
These terms signal a dislike or rejection:
*   **Korean:** 싫어
*   **Chinese:** 可恶
*   **English:** bad, hate, dislike, worst, terrible, awful, horrible, disgusting, annoying

## Detection Algorithm

The detection logic is straightforward. The system converts the input text to lowercase and counts how many keywords from each list appear in the string.

1.  Count occurrences of `POSITIVE_KEYWORDS`.
2.  Count occurrences of `NEGATIVE_KEYWORDS`.
3.  Compare the totals.

If the positive count is higher, the sentiment is "positive". If the negative count is higher, it's "negative". If the counts are equal or no keywords are found, the system defaults to "neutral".

## Limitations

The current keyword-based system has several known weaknesses:
*   **Sarcasm:** Phrases like "Oh, great, another bug" will be marked as positive because of the word "great".
*   **Context:** Some words change meaning based on the surrounding text.
*   **Mixed Sentiment:** A single sentence expressing both likes and dislikes may result in a neutral score if the keyword counts balance out.
*   **Negation:** "I don't like this" might be misidentified if the system only looks for "like" without understanding the "don't" prefix.

## Future Directions

We plan to improve sentiment detection while maintaining the system's performance. Potential upgrades include:
*   **LLM-Augmented Scoring:** Using a small, fast model to verify sentiment when keyword counts are close or ambiguous.
*   **Multi-dimensional Sentiment:** Moving beyond a simple positive/negative binary to track specific emotions like frustration, excitement, or confusion.
*   **Negation Handling:** Improving the algorithm to recognize common negation patterns in English and Korean.
