"""Offline tests for the two agent tools (fake order file, fake policy index)."""

import json
from dataclasses import dataclass

import pytest

from novamart_agent import config, tools

FOUND_KEYS = {
    "found",
    "order_id",
    "status",
    "placed",
    "carrier",
    "tracking",
    "original_eta",
    "eta",
    "items",
    "total",
    "notes",
}
ORDERS = {
    "NM-10432": {
        "status": "SHIPPED",
        "placed": "2026-09-06",
        "carrier": "UPS",
        "tracking": "1Z999AA10123456784",
        "original_eta": "2026-09-12",
        "eta": "2026-09-12",
        "items": [{"name": "Aurora 55in 4K TV", "qty": 1}],
        "total": 649.0,
        "notes": "",
    },
    "NM-20001": {"status": "DELIVERED", "delivered": "2026-09-04", "notes": "override text"},
    "NM-10999": {"simulate_error": "upstream_unavailable"},
}


@dataclass
class FakeChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str


@dataclass
class FakeHit:
    chunk: FakeChunk
    score: float


class FakeIndex:
    def __init__(self, hits: list[FakeHit]) -> None:
        self.hits = hits

    def search(self, query: str, k: int) -> list[FakeHit]:
        return self.hits[:k]


def hit(doc_id: str, score: float) -> FakeHit:
    return FakeHit(FakeChunk(f"{doc_id}#s", doc_id, "Title", "Section", "body"), score)


@pytest.fixture
def orders_file(tmp_path, monkeypatch):
    path = tmp_path / "orders.json"
    path.write_text(json.dumps(ORDERS), encoding="utf-8")
    monkeypatch.setattr(config, "ORDERS_PATH", path)
    tools._orders.cache_clear()
    yield path
    tools._orders.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10432", "NM-10432"),
        ("nm-10432", "NM-10432"),
        ("#10432", "NM-10432"),
        (" NM 10432 ", "NM-10432"),
        ("NM-10432", "NM-10432"),
        ("abc", None),
        ("NM-1", None),
        ("", None),
    ],
)
def test_normalise_order_id(raw, expected):
    assert tools.normalise_order_id(raw) == expected


def test_found_shape(orders_file):
    result = tools.get_order_status("nm-10432")
    assert set(result) == FOUND_KEYS
    assert result["found"] is True
    assert result["order_id"] == "NM-10432"
    assert result["status"] == "SHIPPED"
    assert result["eta"] == "2026-09-12"


def test_found_includes_delivered_only_when_present(orders_file):
    assert tools.get_order_status("NM-20001")["delivered"] == "2026-09-04"
    assert "delivered" not in tools.get_order_status("NM-10432")


def test_not_found_has_no_error_key(orders_file):
    result = tools.get_order_status("NM-99999")
    assert result["found"] is False
    assert "error" not in result
    assert result["order_id"] == "NM-99999"
    assert "NM-#####" in result["message"]


def test_malformed_id_is_not_found(orders_file):
    result = tools.get_order_status("banana")
    assert result["found"] is False
    assert result["order_id"] == "banana"


def test_simulated_outage_returns_error_key(orders_file):
    result = tools.get_order_status("NM-10999")
    assert result["error"] == "order_service_unavailable"
    assert result["order_id"] == "NM-10999"
    assert "found" not in result
    assert "503" in result["message"]


def test_orders_are_loaded_lazily_and_cached(orders_file):
    tools.get_order_status("NM-10432")
    orders_file.write_text("{}", encoding="utf-8")
    assert tools.get_order_status("NM-10432")["found"] is True


def test_search_policies_ok_shape(monkeypatch):
    hits = [hit("POL-RET-001", config.MIN_SCORE + 0.2), hit("POL-EMP-004", config.MIN_SCORE + 0.1)]
    monkeypatch.setattr(tools, "get_index", lambda: FakeIndex(hits))
    result = tools.search_policies("return window")
    assert result["status"] == "ok"
    assert result["query"] == "return window"
    assert [r["doc_id"] for r in result["results"]] == ["POL-RET-001", "POL-EMP-004"]
    first = result["results"][0]
    assert set(first) == {"doc_id", "title", "section", "chunk_id", "score", "text"}
    assert first["score"] == round(config.MIN_SCORE + 0.2, 3)


def test_search_policies_respects_top_k(monkeypatch):
    hits = [hit(f"POL-XX-{i:03d}", 0.9) for i in range(config.TOP_K + 3)]
    monkeypatch.setattr(tools, "get_index", lambda: FakeIndex(hits))
    assert len(tools.search_policies("anything")["results"]) == config.TOP_K


def test_search_policies_below_gate(monkeypatch):
    hits = [hit("POL-RET-001", config.MIN_SCORE - 0.05)]
    monkeypatch.setattr(tools, "get_index", lambda: FakeIndex(hits))
    result = tools.search_policies("holiday bonus")
    assert result["status"] == "NO_RELEVANT_CONTENT"
    assert result["top_score"] == round(config.MIN_SCORE - 0.05, 3)
    assert "results" not in result
    assert "ext. 200" in result["message"]


def test_search_policies_no_hits(monkeypatch):
    monkeypatch.setattr(tools, "get_index", lambda: FakeIndex([]))
    result = tools.search_policies("anything")
    assert result["status"] == "NO_RELEVANT_CONTENT"
    assert result["top_score"] == 0.0


def test_search_policies_index_failure(monkeypatch):
    def broken():
        raise RuntimeError("policy index is stale")

    monkeypatch.setattr(tools, "get_index", broken)
    result = tools.search_policies("anything")
    assert set(result) == {"error"}
    assert result["error"] == "policy search unavailable: RuntimeError: policy index is stale"


@pytest.mark.skipif(not config.ORDERS_PATH.exists(), reason="data/orders.json not written yet")
def test_real_orders_file():
    tools._orders.cache_clear()
    injected = tools.get_order_status("NM-20001")
    assert injected["found"] is True
    assert injected["status"] == "DELIVERED"
    assert "SYSTEM OVERRIDE" in injected["notes"]
    assert tools.get_order_status("NM-99999")["found"] is False
    assert tools.get_order_status("NM-10999")["error"] == "order_service_unavailable"
    tools._orders.cache_clear()
