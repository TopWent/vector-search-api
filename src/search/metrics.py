from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNTER = Counter(
    "search_requests_total",
    "Total requests by endpoint and status",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "search_request_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

ENCODE_LATENCY = Histogram(
    "search_encode_seconds",
    "Time spent in encode() per batch",
    ["batch_size_bucket"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

INDEX_SIZE = Counter(
    "search_documents_indexed_total",
    "Cumulative count of documents added to the index",
)


def expose() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
