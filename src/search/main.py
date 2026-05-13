import logging
import sys

import uvicorn

from . import logging_setup
from .app import create_app
from .config import load_config
from .embedder import SentenceTransformerEmbedder
from .index import VectorIndex
from .store import DocumentStore


def run() -> None:
    try:
        cfg = load_config()
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)

    logging_setup.configure(cfg.log_level)
    log = logging.getLogger("search")

    log.info("loading model: %s", cfg.model_name)
    embedder = SentenceTransformerEmbedder(cfg.model_name, batch_size=cfg.batch_size)
    log.info("model ready: dim=%d", embedder.dim)

    index = VectorIndex.load(cfg.index_path, dim=embedder.dim)
    store = DocumentStore.load(cfg.store_path)
    log.info("restored: index=%d store=%d", index.size, len(store))

    app = create_app(embedder, index, store, cfg)

    host, _, port = cfg.http_addr.rpartition(":")
    uvicorn.run(app, host=host or "0.0.0.0", port=int(port), log_config=None)


if __name__ == "__main__":
    run()
