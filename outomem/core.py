from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from outomem.layers import DEFAULT_VECTOR_DIM, LayerManager
from outomem.neo4j_layers import Neo4jLayerManager
from outomem.prompts import (
    get_consolidation_prompt,
    get_context_synthesis_prompt,
    get_extraction_prompt,
    get_retrieval_judgment_prompt,
    get_retrieval_plan_prompt,
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
        embed_dim: int = DEFAULT_VECTOR_DIM,
    ) -> None:
        self._provider = create_provider(provider, base_url, api_key, model)
        self._embed_dim = embed_dim
        embed_fn = self._create_api_embed_fn(embed_api_url, embed_api_key, embed_model)
        self._lancedb = LayerManager(db_path, embed_fn=embed_fn, vector_dim=embed_dim)
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

    def _get_existing_memories_summary(self) -> str:
        all_pers = self._lancedb.get_all_personalizations()
        all_lt = self._lancedb.get_all_long_term()
        lines = []
        for p in all_pers[:10]:
            if p.get("is_active", True):
                lines.append(f"- [personal] {p.get('content', '')}")
        for lt in all_lt[:5]:
            lines.append(f"- [factual] {lt.get('content', '')}")
        return "\n".join(lines) if lines else "(no existing memories)"

    def remember(self, conversation: list[dict[str, str]] | str) -> None:
        conv_list = format_conversation(conversation)
        if not conv_list:
            return

        # Check if raw format detected (needs LLM parsing)
        raw_entries = [e for e in conv_list if e.get("role") == "raw"]
        if raw_entries:
            conv_list = self._llm_parse_conversation(raw_entries[0]["content"])

        conv_text = self._format_conv_for_llm(conv_list)

        existing_memories = self._get_existing_memories_summary()
        sys_prompt, user_prompt = get_extraction_prompt(
            conv_text, self._style, existing_memories
        )
        raw_response = self._provider.complete(user_prompt, sys_prompt)
        parsed = safe_json_parse(raw_response)
        if not isinstance(parsed, dict):
            return

        session_id = self._lancedb._generate_session_id()

        personal = parsed.get("personal", [])
        factual = parsed.get("factual", [])
        temporal = parsed.get("temporal", [])
        do_not_store = parsed.get("do_not_store", [])

        duplicates_found = False
        personal_contents = []

        for fact_item in personal:
            if isinstance(fact_item, dict):
                content = fact_item.get("content", "").strip()
                if not content:
                    continue
                intensity = fact_item.get("emotional_intensity", "medium")
                is_contradiction = fact_item.get("is_contradiction", False)
            elif isinstance(fact_item, str):
                content = fact_item.strip()
                if not content:
                    continue
                intensity = "medium"
                is_contradiction = False
            else:
                continue

            personal_contents.append(content)
            sentiment = self._detect_sentiment(content)

            intensity_boost_map = {"high": 0.25, "medium": 0.15, "low": 0.08}
            boost = intensity_boost_map.get(intensity, 0.15)

            similar = self._lancedb.find_active_similar_personalizations(
                content, threshold=0.85
            )
            if similar:
                existing = similar[0]
                if is_contradiction or self._is_contradictory(
                    existing["sentiment"], sentiment
                ):
                    existing_active = (
                        self._lancedb.find_active_similar_personalizations(
                            content, threshold=0.95
                        )
                    )
                    if existing_active:
                        self._lancedb.deactivate_personalization(existing["id"])
                        self._neo4j.deactivate_personalization(existing["id"])
                        change_event_content = (
                            f"취향 변화: {existing['content']} → {content}"
                        )
                        self._lancedb.add_temporal(
                            session_id=session_id,
                            event_type="preference_change",
                            content=change_event_content,
                            metadata="{}",
                            related_personalization_id=existing["id"],
                            old_content=existing["content"],
                            new_content=content,
                        )
                        self._neo4j.add_temporal(
                            session_id=session_id,
                            event_type="preference_change",
                            content=change_event_content,
                            metadata="{}",
                            related_personalization_id=existing["id"],
                            old_content=existing["content"],
                            new_content=content,
                        )
                        self._lancedb.add_personalization(
                            content=content,
                            category="preference",
                            strength=0.8,
                            sentiment=sentiment,
                            contradiction_with=existing["id"],
                        )
                        fact_vector = self._compute_embedding(content)
                        self._neo4j.add_personalization(
                            content=content,
                            category="preference",
                            strength=0.8,
                            sentiment=sentiment,
                            contradiction_with=existing["id"],
                            vector=fact_vector,
                        )
                        duplicates_found = True
                    else:
                        self._lancedb.boost_personalization_strength(
                            existing["id"], boost=boost
                        )
                        self._neo4j.boost_personalization_strength(
                            existing["id"], boost=boost
                        )
                else:
                    self._lancedb.boost_personalization_strength(
                        existing["id"], boost=boost
                    )
                    self._neo4j.boost_personalization_strength(
                        existing["id"], boost=boost
                    )
                    duplicates_found = True
            else:
                self._lancedb.add_personalization(
                    content=content,
                    category="preference",
                    strength=1.0,
                    sentiment=sentiment,
                )
                fact_vector = self._compute_embedding(content)
                self._neo4j.add_personalization(
                    content=content,
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

        all_facts_for_raw = (
            [f for f in temporal]
            + [f for f in factual if isinstance(f, str)]
            + personal_contents
        )
        for fact in all_facts_for_raw:
            if isinstance(fact, str) and fact.strip():
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

    def _llm_parse_conversation(self, raw_text: str) -> list[dict[str, str]]:
        prompt = f"""Parse this text into a conversation format. Identify which parts are:
- USER: user's messages (actual human input)
- ASSISTANT: AI/agent's responses

Original text:
{raw_text}

Output ONLY valid JSON array:
[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]

Rules:
- This tool call is from an AGENT, so AI responses in the text should be marked as "assistant"
- Identify actual user input even without explicit markers
- Preserve original meaning and complete thoughts
- Return ONLY the JSON array, no explanation"""

        response = self._provider.complete(prompt, "")
        parsed = safe_json_parse(response)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
        return [{"role": "user", "content": raw_text}]

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

    def _llm_plan_retrieval(
        self,
        query: str,
    ) -> dict[str, Any]:
        sys_p, usr_p = get_retrieval_plan_prompt(query)
        raw_response = self._provider.complete(usr_p, sys_p)
        parsed = safe_json_parse(raw_response)
        if isinstance(parsed, dict):
            return parsed
        return {
            "intent": "general",
            "layers_to_search": {
                "personalization": query,
                "long_term": query,
                "temporal_sessions": query,
                "raw_facts": query,
            },
            "reasoning": "fallback to full search",
        }

    def _llm_filter_memories(
        self,
        query: str,
        pers_results: list[dict[str, Any]],
        lt_results: list[dict[str, Any]],
        temp_results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        pers_text = self._format_memories_list(pers_results, with_strength=True)
        lt_text = self._format_memories_list(lt_results, with_strength=False)
        temp_text = self._format_events_list(temp_results)

        all_memories = f"PERSONALIZATION:\n{pers_text}\n\nLONG_TERM:\n{lt_text}\n\nTEMPORAL:\n{temp_text}"

        sys_p, usr_p = get_retrieval_judgment_prompt(query, all_memories)
        raw_response = self._provider.complete(usr_p, sys_p)
        parsed = safe_json_parse(raw_response)

        if not isinstance(parsed, dict):
            return pers_results, lt_results, temp_results

        selected = parsed.get("selected_memories", [])
        selected_ids = {s.get("id") for s in selected if isinstance(s, dict)}

        if not selected_ids:
            return [], [], []

        def filter_by_id(results, ids):
            return [r for r in results if r.get("id") in ids]

        return (
            filter_by_id(pers_results, selected_ids),
            filter_by_id(lt_results, selected_ids),
            filter_by_id(temp_results, selected_ids),
        )

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

        plan = self._llm_plan_retrieval(conv_text)
        layers_to_search = plan.get("layers_to_search", {})

        pers_results: list[dict[str, Any]] = []
        lt_results: list[dict[str, Any]] = []
        temp_results: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []

        if layers_to_search.get("personalization"):
            pers_query = layers_to_search["personalization"]
            pers_embedding = self._compute_embedding(pers_query)
            pers_results = self._lancedb.search(
                "personalization", pers_embedding, limit=5
            )
            pers_results.sort(key=lambda r: r.get("_distance", float("inf")))

        if layers_to_search.get("long_term"):
            lt_query = layers_to_search["long_term"]
            lt_embedding = self._compute_embedding(lt_query)
            lt_results = self._lancedb.search("long_term", lt_embedding, limit=5)
            lt_results.sort(key=lambda r: r.get("_distance", float("inf")))

        if layers_to_search.get("temporal_sessions"):
            temp_query = layers_to_search["temporal_sessions"]
            temp_embedding = self._compute_embedding(temp_query)
            temp_results = self._lancedb.search(
                "temporal_sessions", temp_embedding, limit=5
            )
            temp_results.sort(key=lambda r: r.get("_distance", float("inf")))

        if layers_to_search.get("raw_facts"):
            raw_query = layers_to_search["raw_facts"]
            raw_embedding = self._compute_embedding(raw_query)
            raw_results = self._lancedb.search("raw_facts", raw_embedding, limit=2)
            raw_results.sort(key=lambda r: r.get("_distance", float("inf")))

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

    def health_check(self) -> dict[str, Any]:
        lancedb_ok = self._lancedb.check_connection()
        tables = self._lancedb.check_tables()
        table_stats = self._lancedb.get_table_stats()
        embedding_ok = self._lancedb.check_embedding()
        neo4j_ok = self._neo4j.check_connection()
        node_counts = self._neo4j.get_node_counts()

        all_tables_ok = all(tables.values())
        overall = lancedb_ok and all_tables_ok and embedding_ok and neo4j_ok

        result: dict[str, Any] = {
            "healthy": overall,
            "lancedb": {
                "connected": lancedb_ok,
                "tables": tables,
                "stats": table_stats,
            },
            "embedding": {"working": embedding_ok},
            "neo4j": {
                "connected": neo4j_ok,
                "node_counts": node_counts,
            },
        }

        if not overall:
            errors: dict[str, Any] = {}
            if not lancedb_ok:
                conn_err = self._lancedb.get_last_connection_error()
                if conn_err:
                    errors["lancedb_connection"] = conn_err
            if not all_tables_ok:
                table_errs = self._lancedb.get_last_table_errors()
                if table_errs:
                    errors["lancedb_tables"] = table_errs
            if not embedding_ok:
                embed_err = self._lancedb.get_last_embedding_error()
                if embed_err:
                    errors["embedding"] = embed_err
            if not neo4j_ok:
                neo4j_err = self._neo4j.get_last_connection_error()
                if neo4j_err:
                    errors["neo4j_connection"] = neo4j_err
            node_errs = self._neo4j.get_last_node_count_errors()
            if node_errs:
                errors["neo4j_queries"] = node_errs
            if errors:
                result["errors"] = errors

        return result

    def export_backup(self, path: str) -> None:
        backup = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "embed_config": {"dimensions": self._embed_dim},
            "lancedb": self._lancedb.export_data(),
            "neo4j": self._neo4j.export_data(),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(backup, file, ensure_ascii=False, indent=2)

    def import_backup(self, path: str, reembed: bool = True) -> None:
        with open(path, encoding="utf-8") as file:
            backup = json.load(file)

        if not isinstance(backup, dict):
            raise ValueError("Backup file must contain a JSON object")
        if backup.get("version") != "1.0":
            raise ValueError(f"Unsupported backup version: {backup.get('version')}")

        backup_dimensions = backup.get("embed_config", {}).get("dimensions")
        if not reembed and backup_dimensions != self._embed_dim:
            raise ValueError(
                "Cannot import without re-embedding when backup dimensions do not match the current embedding configuration"
            )

        embed_fn = self._lancedb._embed_fn
        self._lancedb.import_data(backup.get("lancedb", {}), embed_fn)
        self._neo4j.import_data(backup.get("neo4j", {}), embed_fn)
