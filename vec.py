"""Ingest a directory of text files and search them from the command line."""

import argparse
from pathlib import Path

from src.engine import VectorEngine


def iter_files(directory):
    """Yield regular files below ``directory`` in deterministic path order."""
    return sorted(path for path in directory.rglob("*") if path.is_file())


def ingest_directory(engine, directory):
    """Embed every UTF-8-readable file and return its assigned IDs."""
    ids = []
    for path in iter_files(directory):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue
        ids.append(engine.add_document(text, metadata={
            "path": str(path),
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }))
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing files to ingest")
    parser.add_argument("--query", help="search once instead of opening an interactive prompt")
    parser.add_argument("--index", choices=("brute", "lsh", "hnsw"), default="brute")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--save", type=Path, help="optional JSON store output path")
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        parser.error(f"directory does not exist or is not a directory: {args.directory}")
    index_type = None if args.index == "brute" else args.index
    index_kwargs = {"seed": 42} if args.index == "hnsw" else ({"seed": 42} if args.index == "lsh" else {})
    engine = VectorEngine(index_type=index_type, index_kwargs=index_kwargs)
    ids = ingest_directory(engine, args.directory)
    print(f"Ingested {len(ids)} files from {args.directory}")

    if args.save:
        engine.save(args.save)
        print(f"Saved vectors to {args.save}")

    def search_and_print(query):
        results = engine.search_text(query, k=args.k)
        if not results:
            print("No matching results.")
            return
        for document, distance in results:
            preview = " ".join(document.split())[:120]
            print(f"{distance:.6f}  {preview}")

    if args.query:
        search_and_print(args.query)
        return 0

    print("Enter a query, or press Ctrl-D/Ctrl-C to exit.")
    try:
        while True:
            search_and_print(input("> "))
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
