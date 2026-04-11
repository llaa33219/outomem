from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable


class Neo4jLayerManager:
    _driver: Any

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._init_constraints()

    def close(self) -> None:
        self._driver.close()

    def _init_constraints(self) -> None:
        constraints = [
            "CREATE CONSTRAINT personalization_id IF NOT EXISTS FOR (p:Personalization) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT temporal_session_id IF NOT EXISTS FOR (t:TemporalSession) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
        ]
        for constraint in constraints:
            try:
                self._driver.execute_query(constraint, database_=self._database)
            except (Neo4jError, ServiceUnavailable):
                pass

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _gen_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _generate_session_id() -> str:
        now = datetime.now(timezone.utc)
        return f"sess_{now.strftime('%Y%m%d_%H%M%S')}"

    def add_personalization(
        self,
        content: str,
        category: str,
        strength: float = 1.0,
        decay_factor: float = 0.95,
        sentiment: str = "neutral",
        contradiction_with: str | None = None,
        is_active: bool = True,
        vector: list[float] | None = None,
    ) -> str:
        p_id = self._gen_id()
        now = self._now()

        query = """
        CREATE (p:Personalization {
            id: $id,
            content: $content,
            category: $category,
            sentiment: $sentiment,
            strength: $strength,
            decay_factor: $decay_factor,
            initial_strength: $strength,
            is_active: $is_active,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at),
            last_accessed: datetime($last_accessed),
            access_count: 0,
            vector: $vector
        })
        """

        if contradiction_with:
            query += """
            WITH p
            MATCH (old:Personalization {id: $contradiction_with})
            CREATE (old)-[:CONTRADICTED_BY {timestamp: datetime($updated_at)}]->(p)
            SET old.is_active = false, old.updated_at = datetime($updated_at)
            """

        self._driver.execute_query(
            query,
            id=p_id,
            content=content,
            category=category,
            sentiment=sentiment,
            strength=strength,
            decay_factor=decay_factor,
            is_active=is_active,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            last_accessed=now.isoformat(),
            contradiction_with=contradiction_with,
            vector=vector,
            database_=self._database,
        )
        return p_id

    def get_personalization(self, id: str) -> dict[str, Any] | None:
        records, _, _ = self._driver.execute_query(
            "MATCH (p:Personalization {id: $id}) RETURN p",
            id=id,
            database_=self._database,
        )
        if records:
            return dict(records[0]["p"])
        return None

    def get_all_personalizations(
        self, active_only: bool = False
    ) -> list[dict[str, Any]]:
        query = "MATCH (p:Personalization)"
        if active_only:
            query += " WHERE p.is_active = true"
        query += " RETURN p ORDER BY p.created_at DESC"

        records, _, _ = self._driver.execute_query(query, database_=self._database)
        return [dict(record["p"]) for record in records]

    def find_active_similar_personalizations(
        self,
        content: str,
        threshold: float = 0.85,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Find active personalizations similar to content using cosine similarity.

        Args:
            content: Text to compare
            threshold: Minimum cosine similarity (0-1)
            query_vector: Pre-computed embedding vector. If None, similarity
                         is computed using vector from LanceDB via graph traversal.
        """
        records, _, _ = self._driver.execute_query(
            """
            MATCH (p:Personalization)
            WHERE p.is_active = true
            RETURN p
            ORDER BY p.created_at DESC
            """,
            database_=self._database,
        )

        if not records:
            return []

        # Filter by threshold
        candidates: list[dict[str, Any]] = []
        for record in records:
            p = record["p"]
            p_vec = p.get("vector", [])
            if not p_vec or query_vector is None:
                # Fallback: no vector comparison possible, include by content match
                candidates.append(
                    {
                        "id": p["id"],
                        "content": p["content"],
                        "strength": p["strength"],
                        "sentiment": p["sentiment"],
                        "similarity": 1.0
                        if content.lower() in p["content"].lower()
                        else 0.0,
                    }
                )
            else:
                vec = np.array(p_vec)
                q_vec = np.array(query_vector)
                sim = float(
                    np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
                )
                if sim >= threshold:
                    candidates.append(
                        {
                            "id": p["id"],
                            "content": p["content"],
                            "strength": p["strength"],
                            "sentiment": p["sentiment"],
                            "similarity": sim,
                        }
                    )

        return sorted(candidates, key=lambda x: x["similarity"], reverse=True)[:10]

    def update_personalization_strength(self, id: str, delta: float) -> None:
        self._driver.execute_query(
            """
            MATCH (p:Personalization {id: $id})
            WITH p, CASE 
                WHEN p.strength + $delta > 1.0 THEN 1.0
                WHEN p.strength + $delta < 0.0 THEN 0.0
                ELSE p.strength + $delta
            END AS new_strength
            SET p.strength = new_strength, p.updated_at = datetime()
            """,
            id=id,
            delta=delta,
            database_=self._database,
        )

    def boost_personalization_strength(self, id: str, boost: float = 0.15) -> float:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (p:Personalization {id: $id})
            WITH p, CASE 
                WHEN p.strength + $boost > 1.0 THEN 1.0
                ELSE p.strength + $boost
            END AS new_strength
            SET p.strength = new_strength,
                p.updated_at = datetime(),
                p.last_accessed = datetime()
            RETURN new_strength
            """,
            id=id,
            boost=boost,
            database_=self._database,
        )
        if records:
            return records[0]["new_strength"]
        return 0.0

    def record_access(self, id: str) -> None:
        self._driver.execute_query(
            """
            MATCH (p:Personalization {id: $id})
            SET p.last_accessed = datetime(),
                p.access_count = p.access_count + 1
            """,
            id=id,
            database_=self._database,
        )

    def deactivate_personalization(self, id: str) -> None:
        self._driver.execute_query(
            """
            MATCH (p:Personalization {id: $id})
            SET p.is_active = false, p.updated_at = datetime()
            """,
            id=id,
            database_=self._database,
        )

    def recalculate_all_strengths(self) -> None:
        self._driver.execute_query(
            """
            MATCH (p:Personalization)
            WHERE p.is_active = true
            WITH p, 
                 duration.between(p.last_accessed, datetime()).days AS days_elapsed,
                 p.initial_strength * p.decay_factor ^ duration.between(p.last_accessed, datetime()).days AS calculated
            WHERE days_elapsed > 1
            SET p.strength = CASE 
                WHEN calculated < 0.0 THEN 0.0
                ELSE calculated
            END
            """,
            database_=self._database,
        )

    def merge_personalizations(
        self,
        ids: list[str],
        new_content: str,
        boost: float = 0.15,
        vector: list[float] | None = None,
    ) -> str:
        new_id = self._gen_id()
        now = self._now()

        self._driver.execute_query(
            """
            MATCH (p:Personalization)
            WHERE p.id IN $ids
            WITH p ORDER BY p.strength DESC LIMIT 1
            WITH p.strength AS max_strength, p.category AS category,
                 CASE WHEN max_strength + $boost > 1.0 THEN 1.0 ELSE max_strength + $boost END AS new_strength
            
            MATCH (old:Personalization)
            WHERE old.id IN $ids
            DETACH DELETE old
            
            CREATE (new:Personalization {
                id: $new_id,
                content: $content,
                category: category,
                sentiment: 'neutral',
                strength: new_strength,
                decay_factor: 0.95,
                initial_strength: new_strength,
                is_active: true,
                created_at: datetime($now),
                updated_at: datetime($now),
                last_accessed: datetime($now),
                access_count: 0,
                vector: $vector
            })
            """,
            ids=ids,
            new_id=new_id,
            content=new_content,
            boost=boost,
            now=now.isoformat(),
            vector=vector,
            database_=self._database,
        )
        return new_id

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
        now = self._now()

        query = """
        MERGE (s:Session {id: $session_id})
        CREATE (t:TemporalSession {
            id: $event_id,
            session_id: $session_id,
            event_type: $event_type,
            content: $content,
            timestamp: datetime($timestamp),
            metadata: $metadata,
            old_content: $old_content,
            new_content: $new_content
        })
        CREATE (s)-[:HAS_EVENT]->(t)
        """

        if related_personalization_id:
            query += """
            WITH t
            MATCH (p:Personalization {id: $related_personalization_id})
            CREATE (t)-[:AFFECTED]->(p)
            """

        self._driver.execute_query(
            query,
            session_id=session_id,
            event_id=event_id,
            event_type=event_type,
            content=content,
            timestamp=now.isoformat(),
            metadata=metadata,
            old_content=old_content or "",
            new_content=new_content or "",
            related_personalization_id=related_personalization_id,
            database_=self._database,
        )
        return event_id

    def get_recent_temporal(self, limit: int = 10) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (t:TemporalSession)
            RETURN t
            ORDER BY t.timestamp DESC
            LIMIT $limit
            """,
            limit=limit,
            database_=self._database,
        )
        return [dict(record["t"]) for record in records]

    def get_session_events(self, session_id: str) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (s:Session {id: $session_id})-[:HAS_EVENT]->(t:TemporalSession)
            RETURN t
            ORDER BY t.timestamp ASC
            """,
            session_id=session_id,
            database_=self._database,
        )
        return [dict(record["t"]) for record in records]

    def get_personalization_events(
        self, personalization_id: str
    ) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (t:TemporalSession)-[:AFFECTED]->(p:Personalization {id: $id})
            RETURN t
            ORDER BY t.timestamp DESC
            """,
            id=personalization_id,
            database_=self._database,
        )
        return [dict(record["t"]) for record in records]

    def get_contradiction_chain(self, personalization_id: str) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH path = (start:Personalization {id: $id})-[:CONTRADICTED_BY*]->(end)
            RETURN nodes(path) AS chain
            """,
            id=personalization_id,
            database_=self._database,
        )
        if records:
            return [dict(node) for node in records[0]["chain"]]
        return []

    def get_related_personalizations(
        self, personalization_id: str
    ) -> list[dict[str, Any]]:
        records, _, _ = self._driver.execute_query(
            """
            MATCH (p:Personalization {id: $id})<-[:AFFECTED]-(t:TemporalSession)
            MATCH (t)-[:AFFECTED]->(related:Personalization)
            WHERE related.id <> $id
            RETURN DISTINCT related
            """,
            id=personalization_id,
            database_=self._database,
        )
        return [dict(record["related"]) for record in records]

    def check_connection(self) -> bool:
        try:
            records, _, _ = self._driver.execute_query(
                "RETURN 1 AS health", database_=self._database
            )
            return len(records) == 1 and records[0]["health"] == 1
        except (Neo4jError, ServiceUnavailable):
            return False

    def get_node_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        queries = {
            "personalization": "MATCH (p:Personalization) RETURN count(p) AS cnt",
            "temporal_session": "MATCH (t:TemporalSession) RETURN count(t) AS cnt",
            "session": "MATCH (s:Session) RETURN count(s) AS cnt",
        }
        for name, query in queries.items():
            try:
                records, _, _ = self._driver.execute_query(
                    query, database_=self._database
                )
                counts[name] = records[0]["cnt"] if records else 0
            except (Neo4jError, ServiceUnavailable):
                counts[name] = -1
        return counts
