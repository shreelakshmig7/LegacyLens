"""
run_ingestion.py
----------------
LegacyLens — RAG System for Legacy Enterprise Codebases — Standalone ingestion CLI
-----------------------------------------------------------------------------------
Standalone ingestion entrypoint for production environments (e.g. Railway). This
script downloads a GitHub ZIP archive for the configured repository, discovers
source files, skips already-ingested file paths from ChromaDB, and processes the
remaining files in idempotent 80-file batches.

Key functions:
    _download_and_extract_repo()
    _already_ingested_paths()
    main()

Author: Shreelakshmi Gopinatha Rao
Project: LegacyLens — RAG System for Legacy Enterprise Codebases
"""

import io
import logging
import os
import pathlib
import sys
import zipfile
from typing import Any, Dict, List, Set

# Load .env from project root so VOYAGE_API_KEY, OPENAI_API_KEY, etc. are set.
try:
    from dotenv import load_dotenv

    _env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

from legacylens.config.constants import CHROMA_PERSIST_DIR, validate_required_env_vars
from legacylens.ingestion.chunker import chunk_file
from legacylens.ingestion.embedder import embed_chunks
from legacylens.ingestion.file_discovery import discover_files
from legacylens.ingestion.reference_scraper import attach_dependencies
from legacylens.retrieval.vector_store import insert_chunks

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

_FILE_BATCH_SIZE: int = 80


def _download_and_extract_repo(repo_owner: str, repo_name: str, ref: str, dest: str) -> None:
    """
    Download a GitHub repo ZIP archive and extract it to the destination path.

    Args:
        repo_owner: GitHub username or organization name.
        repo_name: GitHub repository name.
        ref: Branch name or commit SHA to download.
        dest: Destination directory where extracted repo should live.

    Returns:
        None

    Raises:
        FileNotFoundError: If extracted ZIP folder cannot be located.
        httpx.HTTPError: If ZIP download fails.
    """
    import httpx

    zip_url = f"https://github.com/{repo_owner}/{repo_name}/archive/{ref}.zip"
    logger.info("[ingest] Downloading %s ...", zip_url)
    with httpx.Client(timeout=300.0, follow_redirects=True) as client:
        response = client.get(zip_url)
        response.raise_for_status()

    logger.info("[ingest] Download complete (%d bytes). Extracting...", len(response.content))
    parent = str(pathlib.Path(dest).parent)
    pathlib.Path(parent).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(parent)

    extracted_name = f"{repo_name}-{ref}"
    extracted_path = pathlib.Path(parent) / extracted_name
    if extracted_path.exists() and not pathlib.Path(dest).exists():
        extracted_path.rename(dest)
        logger.info("[ingest] Extracted to %s", dest)
    elif pathlib.Path(dest).exists():
        logger.info("[ingest] Destination %s already exists — skipping rename.", dest)
    else:
        raise FileNotFoundError(f"Extracted path not found: {extracted_path}")


def _already_ingested_paths() -> Set[str]:
    """
    Return already-ingested file paths from ChromaDB metadata.

    Args:
        None

    Returns:
        set[str]: Unique file_path values currently present in ChromaDB.

    Raises:
        None
    """
    try:
        from legacylens.retrieval.vector_store import _get_collection

        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return set()
        raw = collection.get(include=["metadatas"], limit=min(total, 200000))
        paths = {(m or {}).get("file_path", "") for m in (raw.get("metadatas") or []) if m}
        logger.info(
            "[ingest] ChromaDB: %d chunks, %d unique file paths already ingested.",
            total,
            len(paths),
        )
        return paths
    except Exception as exc:
        logger.warning("[ingest] Could not query existing paths (treating as empty): %s", exc)
        return set()


def main() -> int:
    """
    Execute idempotent ingestion in 80-file batches and store chunks in ChromaDB.

    Args:
        None

    Returns:
        int: Process exit code (0 for success, 1 for failure).

    Raises:
        None
    """
    try:
        repo_owner = os.getenv("REPO_OWNER", "shreelakshmig7")
        repo_name = os.getenv("REPO_NAME", "gnucobol-contrib")
        repo_ref = os.getenv("REPO_COMMIT", "").strip() or os.getenv("REPO_BRANCH", "master")

        repo_path_env = os.getenv("REPO_PATH", "").strip()
        clone_dest = repo_path_env or str(pathlib.Path("/data") / repo_name)

        logger.info("[ingest] ChromaDB persist dir: %s", CHROMA_PERSIST_DIR)
        logger.info("[ingest] Repo target path: %s", clone_dest)
        logger.info("[ingest] Repo source: %s/%s @ %s", repo_owner, repo_name, repo_ref)

        if not pathlib.Path(clone_dest).exists():
            _download_and_extract_repo(repo_owner, repo_name, repo_ref, clone_dest)
        else:
            logger.info("[ingest] Repo already at %s — skipping download.", clone_dest)

        os.environ["REPO_PATH"] = clone_dest

        disc = discover_files(clone_dest)
        if not disc.get("success"):
            raise RuntimeError(f"File discovery failed: {disc.get('error')}")
        all_files = disc["data"]["files"]
        logger.info("[ingest] Discovered %d files.", len(all_files))

        ingested = _already_ingested_paths()
        remaining = [f for f in all_files if f not in ingested]
        logger.info(
            "[ingest] To process: %d files (%d already done).",
            len(remaining),
            len(ingested),
        )

        total_stored = 0
        n_batches = max(1, (len(remaining) + _FILE_BATCH_SIZE - 1) // _FILE_BATCH_SIZE)

        for idx in range(n_batches):
            batch = remaining[idx * _FILE_BATCH_SIZE : (idx + 1) * _FILE_BATCH_SIZE]
            logger.info("[ingest] Batch %d/%d — %d files...", idx + 1, n_batches, len(batch))

            chunks: List[Dict[str, Any]] = []
            for file_path in batch:
                chunk_result = chunk_file(file_path)
                if chunk_result.get("success"):
                    chunks.extend(chunk_result["data"]["chunks"])

            if not chunks:
                logger.info("[ingest] Batch %d: no chunks — skipping.", idx + 1)
                continue

            dep = attach_dependencies(chunks)
            if dep.get("success"):
                chunks = dep["data"]["chunks"]

            emb = embed_chunks(chunks)
            if not emb.get("success"):
                logger.error("[ingest] Batch %d: embed failed — %s", idx + 1, emb.get("error"))
                continue
            embedded = emb["data"]["chunks"]

            ins = insert_chunks(embedded, clone_dest)
            if ins.get("success"):
                inserted_count = ins["data"]["inserted_count"]
                total_stored += inserted_count
                logger.info(
                    "[ingest] Batch %d/%d: +%d chunks (total: %d).",
                    idx + 1,
                    n_batches,
                    inserted_count,
                    total_stored,
                )
            else:
                logger.error("[ingest] Batch %d: insert failed — %s", idx + 1, ins.get("error"))

        logger.info("[ingest] === COMPLETE — %d total chunks stored ===", total_stored)
        return 0
    except Exception as exc:
        logger.exception("[ingest] Ingestion failed: %s", exc)
        return 1


if __name__ == "__main__":
    if not os.getenv("REPO_COMMIT") and os.getenv("REPO_BRANCH"):
        os.environ["REPO_COMMIT"] = os.getenv("REPO_BRANCH", "")

    env_check = validate_required_env_vars()
    if not env_check["success"]:
        missing = env_check.get("missing", [])
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")

    raise SystemExit(main())
