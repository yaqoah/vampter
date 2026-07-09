"""
ingestion.run_ingestion
=======================
CLI entrypoint for the Vampter ingestion pipeline.

Usage
-----
Run from the project root with the virtual environment activated:

.. code-block:: bash

    python -m ingestion.run_ingestion [OPTIONS]

Options
-------
--api-url, -a     API endpoint for source documents.
                  Default: ``https://api.example-ota.org/v1/documents``

--storage, -s     Local directory to persist the StorageContext.
                  Default: ``./storage``

--embed, -e       Embedding model URI.
                  Default: ``local:BAAI/bge-small-en-v1.5``

--chunk-size      Token budget per chunk (SentenceSplitter).
                  Default: ``512``

--chunk-overlap   Sliding overlap window in tokens.
                  Default: ``64``

--dry-run         Load & parse only; skip store writes.
                  Default: ``False``

--log-level       Python log level (DEBUG, INFO, WARNING, ERROR).
                  Default: ``INFO``

Exit codes
----------
0   Pipeline completed successfully.
1   No documents were found.
2   Pipeline raised an unexpected exception.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from config import settings
from ingestion.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """Configure root logger with a structured, coloured console handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.run_ingestion",
        description=(
            "Vampter asynchronous OTA-policy document ingestion pipeline.\n"
            "Fetches JSON archives from the OTA API, encodes embeddings into Qdrant,\n"
            "and extracts legal property triples into Neo4j."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--api-url", "-a",
        default="https://api.opentermsarchive.org/v1",
        metavar="URL",
        help="Target API URL for source JSON payloads.",
    )
    parser.add_argument(
        "--storage", "-s",
        default="storage",
        metavar="DIR",
        help="Directory to persist the LlamaIndex StorageContext.",
    )
    parser.add_argument(
        "--embed", "-e",
        default="local:BAAI/bge-small-en-v1.5",
        metavar="URI",
        help="Embedding model URI (e.g. 'local:BAAI/bge-small-en-v1.5').",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        metavar="N",
        help="Token budget per chunk for SentenceSplitter.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        metavar="N",
        help="Sliding overlap window in tokens.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse documents without writing to stores.",
    )
    parser.add_argument(
        "--skip-schema-migration",
        action="store_true",
        default=False,
        help="Skip Neo4j schema migration (useful if already run).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity level.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    """Async main — returns a shell exit code."""
    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    _SEP = "=" * 59
    logger.info(_SEP)
    logger.info("  VAMPTER - Asynchronous Ingestion Pipeline")
    logger.info(_SEP)
    logger.info("  API URL       : %s", args.api_url)
    logger.info("  Storage dir   : %s", Path(args.storage).resolve())
    logger.info("  Embed model   : %s", args.embed)
    logger.info("  Chunk size    : %d tokens", args.chunk_size)
    logger.info("  Chunk overlap : %d tokens", args.chunk_overlap)
    logger.info("  Dry run       : %s", args.dry_run)
    logger.info("  Skip schema   : %s", args.skip_schema_migration)
    logger.info(_SEP)

    try:
        result = await run_pipeline(
            settings=settings,
            api_url=args.api_url,
            storage_dir=args.storage,
            embed_model_uri=args.embed,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=args.dry_run,
            run_schema_migration=not args.skip_schema_migration,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline raised an unexpected exception: %s", exc, exc_info=True)
        return 2

    if result.documents_loaded == 0:
        logger.error(
            "No documents were successfully fetched from URL '%s'.", args.api_url
        )
        logger.error(
            "Ensure the API is reachable and returning a valid JSON list."
        )
        return 1

    logger.info(_SEP)
    logger.info("  INGESTION SUMMARY")
    logger.info(_SEP)
    logger.info("  Documents loaded : %d", result.documents_loaded)
    logger.info("  Nodes parsed     : %d", result.nodes_parsed)
    logger.info("  Neo4j rows       : %d", result.neo4j_rows_inserted)
    logger.info("  Qdrant vectors   : %d", result.qdrant_vectors_stored)
    logger.info("  Elapsed time     : %.2f s", result.elapsed_seconds)
    if result.storage_path:
        logger.info("  Storage persisted: %s", result.storage_path)
    logger.info(_SEP)

    return 0


if __name__ == "__main__":
    parser = _build_argument_parser()
    args = parser.parse_args()
    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)