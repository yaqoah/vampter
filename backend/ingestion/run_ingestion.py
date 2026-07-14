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
        "--incremental",
        action="store_true",
        default=False,
        help="Skip platforms that already have Document data in Neo4j.",
    )
    parser.add_argument(
        "--max-platforms",
        type=int,
        default=None,
        metavar="N",
        help="Limit ingestion to first N platforms (useful for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of platforms to process per batch (for chunked ingestion).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from last checkpoint, skip already completed platforms.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log verbosity level.",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        default=False,
        help="Clear all Neo4j and Qdrant data before ingestion (fresh start).",
    )
    parser.add_argument(
        "--clear-platform",
        metavar="NAME",
        help="Clear data for specific platform THEN continue with ingestion (re-process).",
    )
    parser.add_argument(
        "--clear-only",
        metavar="NAME",
        help="Clear data for specific platform and exit WITHOUT re-ingesting. Use this to remove seed data cleanly.",
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
    logger.info("  API URL        : %s", args.api_url)
    logger.info("  Storage dir    : %s", Path(args.storage).resolve())
    logger.info("  Embed model    : %s", args.embed)
    logger.info("  Chunk size     : %d tokens", args.chunk_size)
    logger.info("  Chunk overlap  : %d tokens", args.chunk_overlap)
    logger.info("  Dry run        : %s", args.dry_run)
    logger.info("  Skip schema    : %s", args.skip_schema_migration)
    logger.info("  Resume         : %s", args.resume)
    logger.info("  Incremental    : %s", args.incremental)
    logger.info("  Max platforms  : %s", args.max_platforms if args.max_platforms else "all")
    logger.info("  Batch size     : %d", args.batch_size)
    if args.clear_only:
        logger.info("  Clear only     : %s (will exit after clearing)", args.clear_only)
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
            incremental=args.incremental,
            max_platforms=args.max_platforms,
            clear_all=args.clear_all,
            clear_platform=args.clear_platform if not args.clear_only else None,
            clear_only=args.clear_only,
            resume=args.resume,
            batch_size=args.batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Pipeline raised an unexpected exception: %s", exc, exc_info=True)
        return 2

    # For clear-only mode, documents_loaded=0 is expected - don't treat as error
    if result.documents_loaded == 0 and not args.clear_only:
        logger.error(
            "No documents were successfully fetched from URL '%s'.", args.api_url
        )
        logger.error(
            "Ensure the API is reachable and returning a valid JSON list."
        )
        return 1
    
    # For clear-only mode, print success message
    if args.clear_only:
        logger.info("Successfully cleared platform '%s' from databases.", args.clear_only)
        return 0

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