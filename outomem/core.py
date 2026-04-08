from __future__ import annotations

from typing import Any

from outomem.layers import LayerManager
from outomem.neo4j_layers import Neo4jLayerManager
from outomem.prompts import (
    get_consolidation_prompt,
    get_context_synthesis_prompt,
    get_extraction_prompt,
)
from outomem.providers import LLMProvider, create_provider
from outomem.utils import (
    count_tokens,
    format_conversation,
    load_style_file,
    safe_json_parse,
    truncate_to_token_limit,
)


POSITIVE_KEYWORDS = (
    "좋아",
    "사랑",
    "싫어",
    "최고",
    "잘",
    "재밌",
    "행복",
    "짱",
    "good",
    "like",
    "love",
    "best",
    "great",
    "awesome",
    "amazing",
    "fantastic",
    "wonderful",
    "excellent",
    "prefer",
)
NEGATIVE_KEYWORDS = (
    "싫어",
    "可恶",
    "bad",
    "hate",
    "dislike",
    "worst",
    "terrible",
    "awful",
    "horrible",
    "disgusting",
    "annoying",
)


class Outomem:
    _provider: LLMProvider
    _lancedb: LayerManager
    _neo4j: Neo4jLayerManager
    _style: str

    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        embed_api_url: str,
        embed_api_key: str,
        embed_model: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        db_path: str = "./outomem.lance",
        style_path: str = "./style.md",
    ) -> None:
        self._provider = create_provider(provider, base_url, api_key, model)
        embed_fn = self._create_api_embed_fn(embed_api_url, embed_api_key, embed_model)
        self._lancedb = LayerManager(db_path, embed_fn=embed_fn)
        self._neo4j = Neo4jLayerManager(
            uri=neo4j_uri,
            auth=(neo4j_user, neo4j_password),
        )
        self._style = load_style_file(style_path)

    def _create_api_embed_fn(self, api_url: str, api_key: str, model: str):
        import requests

        def embed_fn(texts: list[str]) -> list[list[float]]:
            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": model},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if "data" in data:
                return [item["embedding"] for item in data["data"]]
            return data.get("embeddings", [[]])

        return embed_fn

    def _detect_sentiment(self, text: str) -> str:
        text_lower = text.lower()
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"

    def _is_contradictory(self, sentiment1: str, sentiment2: str) -> bool:
        return (
            sentiment1 != "neutral"
            and sentiment2 != "neutral"
            and sentiment1 != sentiment2
        )

    def _recalculate_strengths(self) -> None:
        self._lancedb.recalculate_all_strengths()
        self._neo4j.recalculate_all_strengths()

    def remember(self, conversation: list[dict[str, str]] | str) -> None:
        conv_list = format_conversation(conversation)
        if not conv_list:
            return

        conv_text = self._format_conv_for_llm(conv_list)

        sys_prompt, user_prompt = get_extraction_prompt(conv_text, self._style)
        raw_response = self._provider.complete(user_prompt, sys_prompt)
        parsed = safe_json_parse(raw_response)
        if not isinstance(parsed, dict):
            return

        session_id = self._lancedb._generate_session_id()

        personal = parsed.get("personal", [])
        factual = parsed.get("factual", [])
        temporal = parsed.get("temporal", [])

        duplicates_found = False

        for fact in personal:
            if not isinstance(fact, str) or not fact.strip():
                continue
            sentiment = self._detect_sentiment(fact)
            similar = self._lancedb.find_active_similar_personalizations(
                fact, threshold=0.85
            )
            if similar:
                existing = similar[0]
                if self._is_contradictory(existing["sentiment"], sentiment):
                    existing_active = (
                        self._lancedb.find_active_similar_personalizations(
                            fact, threshold=0.95
                        )
                    )
                    if existing_active:
                        self._lancedb.deactivate_personalization(existing["id"])
                        self._neo4j.deactivate_personalization(existing["id"])
                        change_event_content = (
                            f"취향 변화: {existing['content']} → {fact}"
                        )
                        self._lancedb.add_temporal(
                            session_id=session_id,
                            event_type="preference_change",
                            content=change_event_content,
                            metadata="{}",
                            related_personalization_id=existing["id"],
                            old_content=existing["content"],
                            new_content=fact,
                        )
                        self._neo4j.add_temporal(
                            session_id=session_id,
                            event_type="preference_change",
                            content=change_event_content,
                            metadata="{}",
                            related_personalization_id=existing["id"],
                            old_content=existing["content"],
                            new_content=fact,
                        )
                        self._lancedb.add_personalization(
                            content=fact,
                            category="preference",
                            strength=0.8,
                            sentiment=sentiment,
                            contradiction_with=existing["id"],
                        )
                        fact_vector = self._compute_embedding(fact)
                        self._neo4j.add_personalization(
                            content=fact,
                            category="preference",
                            strength=0.8,
                            sentiment=sentiment,
                            contradiction_with=existing["id"],
                            vector=fact_vector,
                        )
                        duplicates_found = True
                    else:
                        self._lancedb.boost_personalization_strength(
                            existing["id"], boost=0.15
                        )
                        self._neo4j.boost_personalization_strength(
                            existing["id"], boost=0.15
                        )
                else:
                    self._lancedb.boost_personalization_strength(
                        existing["id"], boost=0.15
                    )
                    self._neo4j.boost_personalization_strength(
                        existing["id"], boost=0.15
                    )
                    duplicates_found = True
            else:
                self._lancedb.add_personalization(
                    content=fact,
                    category="preference",
                    strength=1.0,
                    sentiment=sentiment,
                )
                fact_vector = self._compute_embedding(fact)
                self._neo4j.add_personalization(
                    content=fact,
                    category="preference",
                    strength=1.0,
                    sentiment=sentiment,
                    vector=fact_vector,
                )

        for fact in factual:
            if not isinstance(fact, str) or not fact.strip():
                continue
            similar = self._lancedb.find_similar_long_term(fact, threshold=0.85)
            if not similar:
                self._lancedb.add_long_term(content=fact, source_facts=[])

        for fact in temporal:
            if not isinstance(fact, str) or not fact.strip():
                continue
            self._lancedb.add_temporal(
                session_id=session_id,
                event_type="event",
                content=fact,
            )
            self._neo4j.add_temporal(
                session_id=session_id,
                event_type="event",
                content=fact,
            )

        for fact in temporal + factual + personal:
            if not isinstance(fact, str) or not fact.strip():
                continue
            self._lancedb.add_raw_fact(content=fact, conversation=conv_text)

        if duplicates_found:
            all_personalizations = self._lancedb.get_all_personalizations()
            memories = [
                {
                    "id": p["id"],
                    "content": p["content"],
                    "strength": p.get("strength", 0),
                }
                for p in all_personalizations
            ]
            sys_p, usr_p = get_consolidation_prompt(memories)
            response = self._provider.complete(usr_p, sys_p)
            result = safe_json_parse(response)
            if isinstance(result, dict):
                for merged in result.get("consolidated", []):
                    if (
                        isinstance(merged, dict)
                        and "original_ids" in merged
                        and "content" in merged
                    ):
                        self._lancedb.merge_personalizations(
                            ids=merged["original_ids"],
                            new_content=merged["content"],
                            boost=merged.get("boost", 0.15),
                        )
                        merged_vector = self._compute_embedding(merged["content"])
                        self._neo4j.merge_personalizations(
                            ids=merged["original_ids"],
                            new_content=merged["content"],
                            boost=merged.get("boost", 0.15),
                            vector=merged_vector,
                        )

    def _format_conv_for_llm(self, conv_list: list[dict[str, str]]) -> str:
        lines = []
        for msg in conv_list:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _compute_embedding(self, text: str) -> list[float]:
        return self._lancedb._compute_embedding(text)

    @staticmethod
    def _build_section(label: str, content: str) -> str:
        return f"=== {label} ===\n{content}\n=== END {label} ==="

    def _format_memories_list(
        self, results: list[dict[str, Any]], with_strength: bool
    ) -> str:
        if not results:
            return ""
        lines = []
        for r in results[:5]:
            content = r.get("content", "")
            if with_strength:
                strength = r.get("strength", 0)
                lines.append(f"- {content} (importance: {strength:.0%})")
            else:
                lines.append(f"- {content}")
        return "\n".join(lines)

    def _format_events_list(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for r in results[:5]:
            event_type = r.get("event_type", "")
            if event_type == "preference_change":
                old = r.get("old_content", "")
                new = r.get("new_content", "")
                ts = r.get("timestamp", "")
                if old and new:
                    lines.append(
                        f"[Preference changed] previous: {old}, current: {new}, date: {ts}"
                    )
                else:
                    lines.append(
                        f"[Preference changed] {r.get('content', '')}, date: {ts}"
                    )
            else:
                ts = r.get("timestamp", "")
                content = r.get("content", "")
                if ts:
                    lines.append(f"[Event] {content}, date: {ts}")
                else:
                    lines.append(f"[Event] {content}")
        return "\n".join(lines)

    def _fallback_context(self, pers: str, lt: str, temp: str, raw: str) -> str:
        parts = []
        if pers:
            parts.append(f"사용자 취향: {pers.replace(chr(10), ', ')}")
        if lt:
            parts.append(f"핵심 사실: {lt.replace(chr(10), ', ')}")
        if temp:
            parts.append(f"최근 사건: {temp.replace(chr(10), ', ')}")
        if raw:
            parts.append(f"참고: {raw.replace(chr(10), ', ')}")
        return ". ".join(parts) if parts else "(기억 없음)"

    def get_context(
        self,
        full_history: list[dict[str, str]] | str | None = None,
        max_tokens: int = 4096,
    ) -> str:
        model = self._provider.model

        if not full_history:
            return ""

        conv_list = format_conversation(full_history)
        conv_text = self._format_conv_for_llm(conv_list)
        query_embedding = self._compute_embedding(conv_text)

        pers_results = self._lancedb.search("personalization", query_embedding, limit=5)
        lt_results = self._lancedb.search("long_term", query_embedding, limit=5)
        raw_results = self._lancedb.search("raw_facts", query_embedding, limit=2)
        temp_results = self._lancedb.search(
            "temporal_sessions", query_embedding, limit=5
        )

        for results in [pers_results, lt_results, raw_results, temp_results]:
            results.sort(key=lambda r: r.get("_distance", float("inf")))

        self._recalculate_strengths()

        for p in pers_results:
            try:
                self._lancedb.record_access(p["id"])
                self._neo4j.record_access(p["id"])
            except Exception:
                pass

        pers_text = self._format_memories_list(pers_results, with_strength=True)
        lt_text = self._format_memories_list(lt_results, with_strength=False)
        temp_text = self._format_events_list(temp_results)
        raw_text = self._format_memories_list(raw_results[:2], with_strength=False)

        if not any([pers_text, lt_text, temp_text, raw_text]):
            return (
                truncate_to_token_limit(self._style, max_tokens, model)
                if self._style
                else ""
            )

        try:
            sys_p, usr_p = get_context_synthesis_prompt(
                conversation=conv_text,
                style=self._style,
                personalization=pers_text,
                long_term=lt_text,
                recent_events=temp_text,
            )
            synthesis = self._provider.complete(usr_p, sys_p).strip()
        except Exception:
            synthesis = self._fallback_context(pers_text, lt_text, temp_text, raw_text)

        context = synthesis

        if count_tokens(context, model) > max_tokens:
            context = truncate_to_token_limit(context, max_tokens, model)

        return context
