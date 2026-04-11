from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from outomem.core import Outomem
from outomem.layers import LayerManager
from outomem.neo4j_layers import Neo4jLayerManager


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
        driver.execute_query.return_value = ([{"health": 1}], None, None)
        yield driver


@pytest.fixture
def mock_embed_fn():
    def embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    return embed


def test_layer_manager_check_connection(mock_lancedb, mock_embed_fn):
    manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
    assert manager.check_connection() is True


def test_layer_manager_check_connection_failure(mock_embed_fn):
    with patch("outomem.layers.lancedb") as mock_lancedb_mod:
        db = MagicMock()
        mock_lancedb_mod.connect.return_value = db
        db.list_tables.return_value.tables = [
            "raw_facts",
            "long_term",
            "personalization",
            "temporal_sessions",
        ]

        manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)

        db.list_tables.side_effect = Exception("Connection failed")
        assert manager.check_connection() is False


def test_layer_manager_check_tables(mock_lancedb, mock_embed_fn):
    table_mock = MagicMock()
    table_mock.count_rows.return_value = 0
    mock_lancedb.open_table.return_value = table_mock

    manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
    tables = manager.check_tables()

    assert all(tables.values())
    assert set(tables.keys()) == {
        "raw_facts",
        "long_term",
        "personalization",
        "temporal_sessions",
    }


def test_layer_manager_get_table_stats(mock_lancedb, mock_embed_fn):
    table_mock = MagicMock()
    table_mock.count_rows.return_value = 42
    mock_lancedb.open_table.return_value = table_mock

    manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
    stats = manager.get_table_stats()

    assert all(v == 42 for v in stats.values())


def test_layer_manager_check_embedding(mock_lancedb, mock_embed_fn):
    manager = LayerManager(db_path="./test.lance", embed_fn=mock_embed_fn)
    assert manager.check_embedding() is True


def test_layer_manager_check_embedding_bad_dim(mock_lancedb):
    bad_embed = lambda texts: [[0.1] * 128 for _ in texts]
    manager = LayerManager(db_path="./test.lance", embed_fn=bad_embed)
    assert manager.check_embedding() is False


def test_neo4j_check_connection(mock_neo4j):
    manager = Neo4jLayerManager(uri="bolt://localhost:7687", auth=("neo4j", "pass"))
    assert manager.check_connection() is True


def test_neo4j_check_connection_failure(mock_neo4j):
    from neo4j.exceptions import ServiceUnavailable

    mock_neo4j.execute_query.side_effect = ServiceUnavailable("Down")
    manager = Neo4jLayerManager(uri="bolt://localhost:7687", auth=("neo4j", "pass"))
    assert manager.check_connection() is False


def test_neo4j_get_node_counts():
    with patch("outomem.neo4j_layers.GraphDatabase") as mock_gdb:
        driver = MagicMock()
        mock_gdb.driver.return_value = driver

        call_count = [0]
        responses = [
            (None, None, None),
            (None, None, None),
            (None, None, None),
            ([{"cnt": 10}], None, None),
            ([{"cnt": 5}], None, None),
            ([{"cnt": 3}], None, None),
        ]

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(responses):
                return responses[idx]
            return ([], None, None)

        driver.execute_query.side_effect = side_effect

        manager = Neo4jLayerManager(uri="bolt://localhost:7687", auth=("neo4j", "pass"))
        counts = manager.get_node_counts()

        assert counts["personalization"] == 10
        assert counts["temporal_session"] == 5
        assert counts["session"] == 3


def test_outomem_health_check_all_ok(mock_lancedb, mock_embed_fn):
    table_mock = MagicMock()
    table_mock.count_rows.return_value = 0
    mock_lancedb.open_table.return_value = table_mock

    with (
        patch("outomem.core.create_provider") as mock_provider,
        patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
    ):
        mock_provider.return_value = MagicMock()
        neo4j_instance = MagicMock()
        mock_neo4j_cls.return_value = neo4j_instance
        neo4j_instance.check_connection.return_value = True
        neo4j_instance.get_node_counts.return_value = {
            "personalization": 10,
            "temporal_session": 5,
            "session": 2,
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

        result = memory.health_check()

        assert result["healthy"] is True
        assert result["lancedb"]["connected"] is True
        assert all(result["lancedb"]["tables"].values())
        assert result["embedding"]["working"] is True
        assert result["neo4j"]["connected"] is True


def test_outomem_health_check_partial_failure(mock_lancedb, mock_embed_fn):
    table_mock = MagicMock()
    table_mock.count_rows.return_value = 0
    mock_lancedb.open_table.return_value = table_mock

    with (
        patch("outomem.core.create_provider") as mock_provider,
        patch("outomem.core.Neo4jLayerManager") as mock_neo4j_cls,
    ):
        mock_provider.return_value = MagicMock()
        neo4j_instance = MagicMock()
        mock_neo4j_cls.return_value = neo4j_instance
        neo4j_instance.check_connection.return_value = False
        neo4j_instance.get_node_counts.return_value = {
            "personalization": -1,
            "temporal_session": -1,
            "session": -1,
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

        result = memory.health_check()

        assert result["healthy"] is False
        assert result["lancedb"]["connected"] is True
        assert result["neo4j"]["connected"] is False
