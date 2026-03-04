"""
clear_chromadb.py
-----------------
Delete the LegacyLens ChromaDB collection so the next ingestion starts fresh.
Uses CHROMA_PERSIST_DIR and collection name from config. Safe to run if the
collection does not exist (e.g. already deleted or never ingested).

Run from project root: PYTHONPATH=. python3 scripts/clear_chromadb.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_COLLECTION_NAME = "legacylens_cobol"


def main() -> None:
    from chromadb.config import Settings
    import chromadb

    from legacylens.config.constants import CHROMA_PERSIST_DIR

    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    names = [c.name for c in client.list_collections()]
    if _COLLECTION_NAME in names:
        client.delete_collection(_COLLECTION_NAME)
        print(f"Deleted collection '{_COLLECTION_NAME}'.")
    else:
        print(f"Collection '{_COLLECTION_NAME}' does not exist (already empty or never ingested).")


if __name__ == "__main__":
    main()
