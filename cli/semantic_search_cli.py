#!/usr/bin/env python3

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    verify_model,
    verify_embeddings,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
)
import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT, pretty_print_results


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    subparsers.add_parser("verify_embeddings", help="Verify the movie embeddings")

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    query_parser = subparsers.add_parser("embedquery", help="Embed a query string")
    query_parser.add_argument("query", type=str, help="The query text to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", type=str, help="The query text to search for")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Limit the number of results",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="The text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="The size of each chunk",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="The overlap between chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text semantically"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=200,
        help="The maximum size of each chunk",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="The overlap between chunks",
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            results = semantic_search(args.query, args.limit)
            pretty_print_results(results)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
