from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, mock_open, patch

import pytest

from outomem.core import Outomem
from outomem.layers import LayerManager
from outomem.neo4j_layers import Neo4jLayerManager


@pytest.fixture
def mock_embed_fn():
    def embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    return embed


@pytest.fixture
def mock_embed_fn_768():
    def embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]

    return embed


@pytest.fixture
def mock_lancedb():
    with patch("outomem.layers.lancedb") as mock:
        db = MagicMock()
        mock.connect.return_value = db
        db.list_tables.return_value.tables = [
            "raw_facts",
            "long_term",
            "personalization",
            "temporal_sessions",
        ]
        yield db


@pytest.fixture
def mock_neo4j():
    with patch("outomem.neo4j_layers.GraphDatabase") as mock:
        driver = MagicMock()
        mock.driver.return_value = driver
        driver.execute_query.return_value = ([], None, None)
        yield driver


class TestLayerManagerExportData:
    def test_export_excludes_vectors(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {
                "id": "test-1",
                "content": "I like dark mode",
                "conversation": "user: I prefer dark",
                "layer": "raw_facts",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "vector": [0.1] * 384,
            }
        ]
        mock_lancedb.open_table.return_value = mock_table

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
        result = manager.export_data()

        assert "raw_facts" in result
        assert len(result["raw_facts"]) == 1
        row = result["raw_facts"][0]
        assert "vector" not in row
        assert row["id"] == "test-1"
        assert row["content"] == "I like dark mode"

    def test_export_all_tables(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_lancedb.open_table.return_value = mock_table

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
        result = manager.export_data()

        assert set(result.keys()) == {
            "raw_facts",
            "long_term",
            "personalization",
            "temporal_sessions",
        }

    def test_export_preserves_metadata(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = [
            {
                "id": "pers-1",
                "content": "likes coffee",
                "category": "preference",
                "sentiment": "positive",
                "layer": "personalization",
                "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
                "strength": 0.85,
                "decay_factor": 0.95,
                "initial_strength": 1.0,
                "last_accessed": datetime(2026, 1, 3, tzinfo=timezone.utc),
                "access_count": 5,
                "contradiction_with": None,
                "is_active": True,
                "vector": [0.1] * 384,
            }
        ]
        mock_lancedb.open_table.return_value = mock_table

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
        result = manager.export_data()

        row = result["personalization"][0]
        assert row["strength"] == 0.85
        assert row["access_count"] == 5
        assert row["is_active"] is True
        assert "vector" not in row


class TestLayerManagerImportData:
    def test_import_reembeds_content(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_lancedb.open_table.return_value = mock_table

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
        data = {
            "raw_facts": [
                {
                    "id": "test-1",
                    "content": "I like dark mode",
                    "conversation": "user: I prefer dark",
                    "layer": "raw_facts",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "long_term": [],
            "personalization": [],
            "temporal_sessions": [],
        }

        manager.import_data(data, mock_embed_fn)

        add_calls = mock_table.add.call_args_list
        assert len(add_calls) == 1
        added_rows = add_calls[0][0][0]
        assert len(added_rows) == 1
        assert "vector" in added_rows[0]
        assert len(added_rows[0]["vector"]) == 384

    def test_import_preserves_strength(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_lancedb.open_table.return_value = mock_table

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
        data = {
            "raw_facts": [],
            "long_term": [],
            "personalization": [
                {
                    "id": "pers-1",
                    "content": "likes coffee",
                    "category": "preference",
                    "sentiment": "positive",
                    "layer": "personalization",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-02T00:00:00+00:00",
                    "strength": 0.75,
                    "decay_factor": 0.95,
                    "initial_strength": 1.0,
                    "last_accessed": "2026-01-03T00:00:00+00:00",
                    "access_count": 3,
                    "is_active": True,
                }
            ],
            "temporal_sessions": [],
        }

        manager.import_data(data, mock_embed_fn)

        calls = mock_table.add.call_args_list
        added_rows = calls[-1][0][0]
        assert added_rows[0]["strength"] == 0.75
        assert added_rows[0]["access_count"] == 3


class TestNeo4jLayerManagerExportData:
    def test_export_excludes_personalization_vectors(self):
        with patch("outomem.neo4j_layers.GraphDatabase") as mock_gdb:
            driver = MagicMock()
            mock_gdb.driver.return_value = driver

            call_count = [0]

            def side_effect(*args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                if idx < 3:
                    return ([], None, None)
                if idx == 3:
                    return (
                        [
                            {
                                "p": {
                                    "id": "pers-1",
                                    "content": "likes coffee",
                                    "category": "preference",
                                    "vector": [0.1] * 384,
                                },
                                "relationships": [
                                    {
                                        "type": "CONTRADICTED_BY",
                                        "target_id": "pers-2",
                                        "timestamp": "2026-01-01T00:00:00Z",
                                    }
                                ],
                            }
                        ],
                        None,
                        None,
                    )
                return ([], None, None)

            driver.execute_query.side_effect = side_effect

            manager = Neo4jLayerManager(
                uri="bolt://localhost:7687", auth=("neo4j", "pass")
            )
            result = manager.export_data()

            assert "personalizations" in result
            pers = result["personalizations"][0]
            assert "vector" not in pers
            assert pers["id"] == "pers-1"
            assert len(pers["relationships"]) == 1
            assert pers["relationships"][0]["type"] == "CONTRADICTED_BY"

    def test_export_preserves_relationships(self):
        with patch("outomem.neo4j_layers.GraphDatabase") as mock_gdb:
            driver = MagicMock()
            mock_gdb.driver.return_value = driver

            call_count = [0]

            def side_effect(*args, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                if idx < 3:
                    return ([], None, None)
                if idx == 3:
                    return ([{"p": {"id": "p1"}, "relationships": []}], None, None)
                if idx == 4:
                    return (
                        [
                            {
                                "t": {"id": "t1", "session_id": "s1"},
                                "relationships": [
                                    {"type": "AFFECTED", "target_id": "p1"}
                                ],
                            }
                        ],
                        None,
                        None,
                    )
                if idx == 5:
                    return (
                        [
                            {
                                "s": {"id": "s1"},
                                "relationships": [
                                    {"type": "HAS_EVENT", "target_id": "t1"}
                                ],
                            }
                        ],
                        None,
                        None,
                    )
                return ([], None, None)

            driver.execute_query.side_effect = side_effect

            manager = Neo4jLayerManager(
                uri="bolt://localhost:7687", auth=("neo4j", "pass")
            )
            result = manager.export_data()

            assert (
                result["temporal_sessions"][0]["relationships"][0]["target_id"] == "p1"
            )
            assert result["sessions"][0]["relationships"][0]["target_id"] == "t1"


class TestNeo4jLayerManagerImportData:
    def test_import_creates_nodes_with_vectors(self):
        with patch("outomem.neo4j_layers.GraphDatabase") as mock_gdb:
            driver = MagicMock()
            mock_gdb.driver.return_value = driver

            def embed(texts):
                return [[0.2] * 384 for _ in texts]

            manager = Neo4jLayerManager(
                uri="bolt://localhost:7687", auth=("neo4j", "pass")
            )

            data = {
                "personalizations": [
                    {
                        "id": "p1",
                        "content": "likes coffee",
                        "category": "preference",
                        "sentiment": "positive",
                        "strength": 1.0,
                        "decay_factor": 0.95,
                        "initial_strength": 1.0,
                        "is_active": True,
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "last_accessed": "2026-01-01T00:00:00Z",
                        "access_count": 0,
                        "relationships": [],
                    }
                ],
                "temporal_sessions": [],
                "sessions": [],
            }

            manager.import_data(data, embed)

            create_calls = [
                call
                for call in driver.execute_query.call_args_list
                if "CREATE (p:Personalization" in str(call)
            ]
            assert len(create_calls) == 1

    def test_import_restores_relationships(self):
        with patch("outomem.neo4j_layers.GraphDatabase") as mock_gdb:
            driver = MagicMock()
            mock_gdb.driver.return_value = driver

            def embed(texts):
                return [[0.2] * 384 for _ in texts]

            manager = Neo4jLayerManager(
                uri="bolt://localhost:7687", auth=("neo4j", "pass")
            )

            data = {
                "personalizations": [
                    {
                        "id": "p1",
                        "content": "likes coffee",
                        "category": "preference",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "relationships": [
                            {
                                "type": "CONTRADICTED_BY",
                                "target_id": "p2",
                                "timestamp": "2026-02-01T00:00:00Z",
                            }
                        ],
                    },
                    {
                        "id": "p2",
                        "content": "hates coffee",
                        "category": "preference",
                        "created_at": "2026-02-01T00:00:00Z",
                        "updated_at": "2026-02-01T00:00:00Z",
                        "relationships": [],
                    },
                ],
                "temporal_sessions": [],
                "sessions": [],
            }

            manager.import_data(data, embed)

            rel_calls = [
                call
                for call in driver.execute_query.call_args_list
                if "CONTRADICTED_BY" in str(call)
            ]
            assert len(rel_calls) == 1


class TestOutomemExportBackup:
    def test_export_writes_json_file(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_lancedb.open_table.return_value = mock_table

        with (
            patch("outomem.core.create_provider") as mock_provider,
            patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
        ):
            mock_provider.return_value = MagicMock()
            neo4j_instance = MagicMock()
            mock_neo4j_cls.return_value = neo4j_instance
            neo4j_instance.export_data.return_value = {
                "personalizations": [],
                "temporal_sessions": [],
                "sessions": [],
            }

            memory = Outomem(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="test",
                model="gpt-4",
                embed_api_url="https://api.openai.com/v1/embeddings",
                embed_api_key="test",
                embed_model="text-embedding-3-small",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="pass",
                db_path="./test.lance",
            )
            memory._lancedb._embed_fn = mock_embed_fn

            m = mock_open()
            with patch("builtins.open", m):
                memory.export_backup("/tmp/backup.json")

            m.assert_called_once_with("/tmp/backup.json", "w", encoding="utf-8")
            written_data = "".join(call.args[0] for call in m().write.call_args_list)
            backup = json.loads(written_data)
            assert backup["version"] == "1.0"
            assert "exported_at" in backup
            assert backup["embed_config"]["dimensions"] == 384
            assert "lancedb" in backup
            assert "neo4j" in backup


class TestOutomemImportBackup:
    def test_import_with_reembed(self, mock_lancedb, mock_embed_fn):
        mock_table = MagicMock()
        mock_table.to_arrow.return_value.to_pylist.return_value = []
        mock_lancedb.open_table.return_value = mock_table

        backup_data = {
            "version": "1.0",
            "exported_at": "2026-04-12T00:00:00+00:00",
            "embed_config": {"dimensions": 384},
            "lancedb": {
                "raw_facts": [],
                "long_term": [],
                "personalization": [],
                "temporal_sessions": [],
            },
            "neo4j": {
                "personalizations": [],
                "temporal_sessions": [],
                "sessions": [],
            },
        }

        with (
            patch("outomem.core.create_provider") as mock_provider,
            patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
        ):
            mock_provider.return_value = MagicMock()
            neo4j_instance = MagicMock()
            mock_neo4j_cls.return_value = neo4j_instance

            memory = Outomem(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="test",
                model="gpt-4",
                embed_api_url="https://api.openai.com/v1/embeddings",
                embed_api_key="test",
                embed_model="text-embedding-3-small",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="pass",
                db_path="./test.lance",
            )
            memory._lancedb._embed_fn = mock_embed_fn

            m = mock_open(read_data=json.dumps(backup_data))
            with patch("builtins.open", m):
                memory.import_backup("/tmp/backup.json", reembed=True)

            assert memory._lancedb._db.open_table.called
            assert neo4j_instance.import_data.called

    def test_import_reembed_false_dimension_mismatch_raises(
        self, mock_lancedb, mock_embed_fn
    ):
        backup_data = {
            "version": "1.0",
            "exported_at": "2026-04-12T00:00:00+00:00",
            "embed_config": {"dimensions": 768},
            "lancedb": {},
            "neo4j": {},
        }

        with (
            patch("outomem.core.create_provider") as mock_provider,
            patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
        ):
            mock_provider.return_value = MagicMock()
            neo4j_instance = MagicMock()
            mock_neo4j_cls.return_value = neo4j_instance

            memory = Outomem(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="test",
                model="gpt-4",
                embed_api_url="https://api.openai.com/v1/embeddings",
                embed_api_key="test",
                embed_model="text-embedding-3-small",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="pass",
                db_path="./test.lance",
            )
            memory._lancedb._embed_fn = mock_embed_fn

            m = mock_open(read_data=json.dumps(backup_data))
            with patch("builtins.open", m):
                with pytest.raises(
                    ValueError, match="Cannot import without re-embedding"
                ):
                    memory.import_backup("/tmp/backup.json", reembed=False)

    def test_import_invalid_version_raises(self, mock_lancedb, mock_embed_fn):
        backup_data = {"version": "2.0"}

        with (
            patch("outomem.core.create_provider") as mock_provider,
            patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
        ):
            mock_provider.return_value = MagicMock()
            neo4j_instance = MagicMock()
            mock_neo4j_cls.return_value = neo4j_instance

            memory = Outomem(
                provider="openai",
                base_url="https://api.openai.com/v1",
                api_key="test",
                model="gpt-4",
                embed_api_url="https://api.openai.com/v1/embeddings",
                embed_api_key="test",
                embed_model="text-embedding-3-small",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="pass",
                db_path="./test.lance",
            )
            memory._lancedb._embed_fn = mock_embed_fn

            m = mock_open(read_data=json.dumps(backup_data))
            with patch("builtins.open", m):
                with pytest.raises(ValueError, match="Unsupported backup version"):
                    memory.import_backup("/tmp/backup.json", reembed=True)
