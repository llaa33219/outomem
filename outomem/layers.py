from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

import numpy as np

import pyarrow as pa
import pyarrow.compute as pc  # type: ignore[reportMissingTypeStubs]

import lancedb

VECTOR_DIM = 384
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


def _default_fastembed_embed(texts: list[str]) -> list[list[float]]:
    """Default local embedding using fastembed."""
    from fastembed import TextEmbedding

    embedder = TextEmbedding(DEFAULT_LOCAL_MODEL)
    return [emb.tolist() for emb in embedder.embed(texts)]


SCHEMAS: dict[str, pa.Schema] = {
    "raw_facts": pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("conversation", pa.utf8()),
            pa.field("layer", pa.utf8()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    ),
    "long_term": pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("source_facts", pa.list_(pa.utf8())),
            pa.field("layer", pa.utf8()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("updated_at", pa.timestamp("us", tz="UTC")),
            pa.field("access_count", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    ),
    "personalization": pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("category", pa.utf8()),
            pa.field("sentiment", pa.utf8()),  # "positive", "negative", "neutral"
            pa.field("layer", pa.utf8()),
            pa.field("created_at", pa.timestamp("us", tz="UTC")),
            pa.field("updated_at", pa.timestamp("us", tz="UTC")),
            pa.field("strength", pa.float64()),
            pa.field("decay_factor", pa.float64()),
            pa.field("initial_strength", pa.float64()),
            pa.field("last_accessed", pa.timestamp("us", tz="UTC")),
            pa.field("access_count", pa.int64()),
            pa.field(
                "contradiction_with", pa.utf8()
            ),  # ID of contradictory fact, if any
            pa.field("is_active", pa.bool_()),  # False if superseded by contradiction
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    ),
    "temporal_sessions": pa.schema(
        [
            pa.field("id", pa.utf8()),
            pa.field("session_id", pa.utf8()),
            pa.field("event_type", pa.utf8()),
            pa.field("content", pa.utf8()),
            pa.field("timestamp", pa.timestamp("us", tz="UTC")),
            pa.field("layer", pa.utf8()),
            pa.field("metadata", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
            pa.field("related_personalization_id", pa.utf8()),
            pa.field("old_content", pa.utf8()),
            pa.field("new_content", pa.utf8()),
        ]
    ),
}


class LayerManager:
    def __init__(
        self,
        db_path: str = "./outomem.lance",
        embed_fn: EmbeddingFunction | None = None,
    ) -> None:
        self._db = lancedb.connect(db_path)
        self._embed_fn = embed_fn or _default_fastembed_embed
        self._init_collections()

    def _compute_embedding(self, text: str) -> list[float]:
        return self._embed_fn([text])[0]

    def _init_collections(self) -> None:
        existing = self._db.list_tables().tables
        for name, schema in SCHEMAS.items():
            if name not in existing:
                self._db.create_table(name, schema=schema)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _generate_session_id() -> str:
        """Auto-generate session ID in format 'sess_YYYYMMDD_HHMMSS' (seconds for uniqueness)."""
        now = datetime.now(timezone.utc)
        return f"sess_{now.strftime('%Y%m%d_%H%M%S')}"

    def _open(self, name: str):
        return self._db.open_table(name)

    def add_raw_fact(self, content: str, conversation: str) -> str:
        fact_id = self._gen_id()
        self._open("raw_facts").add(
            [
                {
                    "id": fact_id,
                    "content": content,
                    "conversation": conversation,
                    "layer": "raw_facts",
                    "created_at": self._now(),
                    "vector": self._compute_embedding(content),
                }
            ]
        )
        return fact_id

    def add_long_term(self, content: str, source_facts: list[str]) -> str:
        lt_id = self._gen_id()
        now = self._now()
        self._open("long_term").add(
            [
                {
                    "id": lt_id,
                    "content": content,
                    "source_facts": source_facts,
                    "layer": "long_term",
                    "created_at": now,
                    "updated_at": now,
                    "access_count": 0,
                    "vector": self._compute_embedding(content),
                }
            ]
        )
        return lt_id

    def update_long_term(self, id: str, content: str) -> None:
        table = self._open("long_term")
        at = table.to_arrow()
        mask = pc.equal(at.column("id"), id)  # pyright: ignore[reportAttributeAccessIssue]
        rows = at.filter(mask).to_pylist()
        if not rows:
            return
        row = rows[0]
        table.delete(f"id = '{id}'")
        row["content"] = content
        row["updated_at"] = self._now()
        row["vector"] = self._compute_embedding(content)
        table.add([row])

    def add_personalization(
        self,
        content: str,
        category: str,
        strength: float = 1.0,
        decay_factor: float = 0.95,
        sentiment: str = "neutral",
        contradiction_with: str | None = None,
        is_active: bool = True,
    ) -> str:
        p_id = self._gen_id()
        now = self._now()
        self._open("personalization").add(
            [
                {
                    "id": p_id,
                    "content": content,
                    "category": category,
                    "sentiment": sentiment,
                    "layer": "personalization",
                    "created_at": now,
                    "updated_at": now,
                    "strength": strength,
                    "decay_factor": decay_factor,
                    "initial_strength": strength,
                    "last_accessed": now,
                    "access_count": 0,
                    "contradiction_with": contradiction_with,
                    "is_active": is_active,
                    "vector": self._compute_embedding(content),
                }
            ]
        )
        return p_id

    def update_personalization_strength(self, id: str, delta: float) -> None:
        table = self._open("personalization")
        at = table.to_arrow()
        mask = pc.equal(at.column("id"), id)  # pyright: ignore[reportAttributeAccessIssue]
        filtered = at.filter(mask)
        if filtered.num_rows == 0:
            return
        current = filtered.column("strength")[0].as_py()
        new_strength = max(0.0, min(1.0, current + delta))
        table.update(
            where=f"id = '{id}'",
            values={"strength": new_strength, "updated_at": self._now()},
        )

    def record_access(self, id: str) -> None:
        table = self._open("personalization")
        table.update(
            where=f"id = '{id}'",
            values={"last_accessed": self._now()},
        )

    def recalculate_and_apply_boost(
        self,
        id: str,
        repeat_count: int,
        base_strength: float,
    ) -> float:
        import math

        table = self._open("personalization")
        at = table.to_arrow()
        mask = pc.equal(at.column("id"), id)  # pyright: ignore[reportAttributeAccessIssue]
        filtered = at.filter(mask)
        if filtered.num_rows == 0:
            return base_strength

        created = filtered.column("created_at")[0].as_py()
        initial = filtered.column("initial_strength")[0].as_py()

        now = self._now()
        days_elapsed = (now - created.replace(tzinfo=timezone.utc)).days
        decay_factor = 0.95
        recalculated = initial * (decay_factor**days_elapsed)
        recalculated = max(0.0, min(1.0, recalculated))

        boost = 0.15 * math.log(1 + repeat_count)
        new_strength = min(1.0, recalculated + boost)
        new_strength = max(0.0, new_strength)

        table.update(
            where=f"id = '{id}'",
            values={
                "strength": new_strength,
                "last_accessed": now,
            },
        )
        return new_strength

    def recalculate_all_strengths(self) -> None:
        table = self._open("personalization")
        at = table.to_arrow()
        if at.num_rows == 0:
            return

        now = self._now()
        one_day_seconds = 86400

        ids_to_update: list[str] = []
        new_strengths: list[float] = []

        for i in range(at.num_rows):
            row_id = at.column("id")[i].as_py()
            created = at.column("created_at")[i].as_py()
            last_accessed = at.column("last_accessed")[i].as_py()
            initial = at.column("initial_strength")[i].as_py()

            if last_accessed is None:
                last_accessed = created

            seconds_elapsed = (
                now - last_accessed.replace(tzinfo=timezone.utc)
            ).total_seconds()

            if seconds_elapsed > one_day_seconds:
                days_elapsed = seconds_elapsed / 86400
                decay_factor = 0.95
                new_strength = initial * (decay_factor**days_elapsed)
                new_strength = max(0.0, min(1.0, new_strength))
                ids_to_update.append(row_id)
                new_strengths.append(new_strength)

        for row_id, new_strength in zip(ids_to_update, new_strengths):
            table.update(
                where=f"id = '{row_id}'",
                values={"strength": new_strength},
            )

    def decay_personalization(self, factor: float) -> None:
        table = self._open("personalization")
        if table.count_rows() == 0:
            return
        table.update(values_sql={"strength": f"strength * {factor}"})

    def merge_personalizations(
        self, ids: list[str], new_content: str, boost: float = 0.15
    ) -> str:
        table = self._open("personalization")
        at = table.to_arrow()
        mask = pc.is_in(at.column("id"), pa.array(ids))  # pyright: ignore[reportAttributeAccessIssue]
        rows = at.filter(mask).to_pylist()

        max_strength = max((r["strength"] for r in rows), default=1.0)
        new_strength = min(1.0, max_strength + boost)
        category = rows[0]["category"] if rows else "preference"

        for id_ in ids:
            table.delete(f"id = '{id_}'")

        return self.add_personalization(
            content=new_content,
            category=category,
            strength=new_strength,
        )

    def add_temporal(
        self,
        session_id: str | None,
        event_type: str,
        content: str,
        metadata: str = "{}",
        related_personalization_id: str | None = None,
        old_content: str | None = None,
        new_content: str | None = None,
    ) -> str:
        if session_id is None:
            session_id = self._generate_session_id()
        event_id = self._gen_id()
        self._open("temporal_sessions").add(
            [
                {
                    "id": event_id,
                    "session_id": session_id,
                    "event_type": event_type,
                    "content": content,
                    "timestamp": self._now(),
                    "layer": "temporal_sessions",
                    "metadata": metadata,
                    "vector": self._compute_embedding(content),
                    "related_personalization_id": related_personalization_id or "",
                    "old_content": old_content or "",
                    "new_content": new_content or "",
                }
            ]
        )
        return event_id

    def search(
        self,
        layer: str,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if layer not in SCHEMAS:
            raise ValueError(f"Unknown layer: {layer}")
        table = self._open(layer)
        if table.count_rows() == 0:
            return []
        return table.search(query_embedding).limit(limit).to_list()

    def get_all_personalizations(self) -> list[dict[str, Any]]:
        table = self._open("personalization")
        return table.to_arrow().to_pylist()

    def get_recent_temporal(self, limit: int = 10) -> list[dict[str, Any]]:
        table = self._open("temporal_sessions")
        at = table.to_arrow()
        if at.num_rows == 0:
            return []
        indices = pc.sort_indices(at, sort_keys=[("timestamp", "descending")])  # pyright: ignore[reportAttributeAccessIssue]
        sorted_at = at.take(indices)
        return sorted_at.slice(0, limit).to_pylist()

    def get_all_long_term(self) -> list[dict[str, Any]]:
        table = self._open("long_term")
        return table.to_arrow().to_pylist()

    def find_similar_personalizations(
        self,
        content: str,
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        query_emb = np.array(self._compute_embedding(content))
        all_pers = self.get_all_personalizations()
        if not all_pers:
            return []

        similar: list[dict[str, Any]] = []
        for p in all_pers:
            p_emb = p.get("vector", [])
            if not p_emb:
                continue
            p_vec = np.array(p_emb)
            sim = float(
                np.dot(query_emb, p_vec)
                / (np.linalg.norm(query_emb) * np.linalg.norm(p_vec))
            )
            if sim >= threshold:
                similar.append(
                    {
                        "id": p["id"],
                        "content": p["content"],
                        "strength": p.get("strength", 0),
                        "similarity": sim,
                    }
                )
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def find_active_similar_personalizations(
        self,
        content: str,
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        query_emb = np.array(self._compute_embedding(content))
        all_pers = self.get_all_personalizations()
        if not all_pers:
            return []

        active_similar: list[dict[str, Any]] = []
        for p in all_pers:
            if not p.get("is_active", True):
                continue
            p_emb = p.get("vector", [])
            if not p_emb:
                continue
            p_vec = np.array(p_emb)
            sim = float(
                np.dot(query_emb, p_vec)
                / (np.linalg.norm(query_emb) * np.linalg.norm(p_vec))
            )
            if sim >= threshold:
                active_similar.append(
                    {
                        "id": p["id"],
                        "content": p["content"],
                        "strength": p.get("strength", 0),
                        "sentiment": p.get("sentiment", "neutral"),
                        "similarity": sim,
                    }
                )
        return sorted(active_similar, key=lambda x: x["similarity"], reverse=True)

    def deactivate_personalization(self, id: str) -> None:
        table = self._open("personalization")
        table.update(
            where=f"id = '{id}'",
            values={"is_active": False, "updated_at": self._now()},
        )

    def boost_personalization_strength(self, id: str, boost: float = 0.15) -> float:
        table = self._open("personalization")
        at = table.to_arrow()
        mask = pc.equal(at.column("id"), id)
        filtered = at.filter(mask)
        if filtered.num_rows == 0:
            return 0.0
        current = filtered.column("strength")[0].as_py()
        new_strength = min(1.0, current + boost)
        table.update(
            where=f"id = '{id}'",
            values={
                "strength": new_strength,
                "updated_at": self._now(),
                "last_accessed": self._now(),
            },
        )
        return new_strength

    def find_similar_long_term(
        self,
        content: str,
        threshold: float = 0.85,
    ) -> list[dict[str, Any]]:
        query_emb = np.array(self._compute_embedding(content))
        all_lt = self.get_all_long_term()
        if not all_lt:
            return []

        similar: list[dict[str, Any]] = []
        for lt in all_lt:
            lt_emb = lt.get("vector", [])
            if not lt_emb:
                continue
            lt_vec = np.array(lt_emb)
            sim = float(
                np.dot(query_emb, lt_vec)
                / (np.linalg.norm(query_emb) * np.linalg.norm(lt_vec))
            )
            if sim >= threshold:
                similar.append(
                    {
                        "id": lt["id"],
                        "content": lt["content"],
                        "similarity": sim,
                    }
                )
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def check_connection(self) -> bool:
        """Check if LanceDB connection is alive."""
        try:
            _ = self._db.list_tables()
            return True
        except Exception:
            return False

    def check_tables(self) -> dict[str, bool]:
        """Check if all required tables exist and are accessible."""
        results: dict[str, bool] = {}
        existing = self._db.list_tables().tables
        for name in SCHEMAS:
            try:
                if name in existing:
                    table = self._open(name)
                    _ = table.count_rows()
                    results[name] = True
                else:
                    results[name] = False
            except Exception:
                results[name] = False
        return results

    def get_table_stats(self) -> dict[str, int]:
        """Get row count for each table."""
        stats: dict[str, int] = {}
        for name in SCHEMAS:
            try:
                table = self._open(name)
                stats[name] = table.count_rows()
            except Exception:
                stats[name] = -1
        return stats

    def check_embedding(self, test_text: str = "health check test") -> bool:
        """Verify embedding function works correctly."""
        try:
            embedding = self._compute_embedding(test_text)
            return (
                isinstance(embedding, list)
                and len(embedding) == VECTOR_DIM
                and all(isinstance(x, float) for x in embedding)
            )
        except Exception:
            return False
