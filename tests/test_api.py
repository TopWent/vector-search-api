def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["indexed"] == 0
    assert body["dim"] > 0


def test_metrics_exposes_prometheus(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "search_requests_total" in resp.text


def test_embeddings_endpoint(client):
    resp = client.post("/embeddings", json={"texts": ["hello", "world"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dim"] > 0
    assert len(body["vectors"]) == 2
    assert len(body["vectors"][0]) == body["dim"]


def test_embeddings_rejects_empty_list(client):
    resp = client.post("/embeddings", json={"texts": []})
    assert resp.status_code == 422


def test_documents_then_search(client):
    upsert = client.post(
        "/documents",
        json={
            "documents": [
                {"id": "1", "text": "machine learning models"},
                {"id": "2", "text": "cooking pasta recipes"},
                {"id": "3", "text": "deep learning research"},
            ],
        },
    )
    assert upsert.status_code == 200
    assert upsert.json() == {"upserted": 3, "total": 3}

    search = client.post("/search", json={"query": "any text", "k": 2})
    assert search.status_code == 200
    hits = search.json()["hits"]
    assert len(hits) == 2
    for hit in hits:
        assert hit["id"] in ("1", "2", "3")
        assert -1.001 <= hit["score"] <= 1.001


def test_search_on_empty_index_returns_empty_hits(client):
    resp = client.post("/search", json={"query": "anything", "k": 5})
    assert resp.status_code == 200
    assert resp.json() == {"hits": []}


def test_upsert_validation(client):
    resp = client.post("/documents", json={"documents": []})
    assert resp.status_code == 422

    resp = client.post(
        "/documents",
        json={
            "documents": [{"id": "", "text": "x"}],
        },
    )
    assert resp.status_code == 422


def test_search_validation(client):
    resp = client.post("/search", json={"query": "", "k": 3})
    assert resp.status_code == 422

    resp = client.post("/search", json={"query": "x", "k": 0})
    assert resp.status_code == 422

    resp = client.post("/search", json={"query": "x", "k": 1000})
    assert resp.status_code == 422


def test_upsert_overwrites_count(client):
    client.post("/documents", json={"documents": [{"id": "1", "text": "first"}]})
    resp = client.post(
        "/documents",
        json={
            "documents": [
                {"id": "1", "text": "first updated"},
                {"id": "2", "text": "second"},
            ],
        },
    )
    assert resp.json() == {"upserted": 1, "total": 2}
