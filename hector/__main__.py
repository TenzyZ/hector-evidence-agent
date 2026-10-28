"""JSON CLI for local ingestion, retrieval and Configuration A evaluation."""

import argparse
from dataclasses import asdict
import glob
import json
from pathlib import Path
import sqlite3
import sys

from . import config
from .evaluate import run, validate
from .ingest import ingest
from .store import open_store


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = Parser(prog="python -m hector")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("ingest")
    add.add_argument("--db", required=True)
    add.add_argument("files", nargs="+")
    search = commands.add_parser("search")
    search.add_argument("--db", required=True)
    search.add_argument("-k", type=int, default=config.SEARCH_K_DEFAULT)
    search.add_argument("query")
    evaluation = commands.add_parser("eval").add_subparsers(dest="evaluation", required=True)
    for command in ("validate", "run"):
        child = evaluation.add_parser(command)
        child.add_argument("--corpus", default="eval/corpus")
        child.add_argument("--questions", default="eval/questions.jsonl")
        if command == "run":
            child.add_argument("--config", choices=("A",), required=True)
            child.add_argument("--split", choices=("dev",), required=True)
    accepted = None
    filename = None
    try:
        args = parser.parse_args(argv)
        if args.command == "ingest":
            accepted = []
            with open_store(args.db) as store:
                for pattern in args.files:
                    filename = pattern
                    paths = sorted(glob.glob(pattern)) if glob.escape(pattern) != pattern else [pattern]
                    if not paths:
                        raise FileNotFoundError("No files match the ingestion pattern.")
                    for filename in paths:
                        with Path(filename).open("rb") as stream:
                            document, chunks = ingest(filename, stream.read(config.MAX_FILE_BYTES + 1))
                        store.add(document, chunks)
                        accepted.append({"doc_id": document.doc_id, "filename": document.filename,
                                         "chunks": len(chunks)})
            result = {"documents": accepted}
        elif args.command == "search":
            with open_store(args.db, create=False) as store:
                result = {"hits": [asdict(hit) for hit in store.search(args.query, args.k)]}
        elif args.evaluation == "validate":
            dataset = validate(args.corpus, args.questions)
            result = {"valid": True, "documents": len(dataset.documents), "items": len(dataset.items),
                      **dataset.identities}
        else:
            result = run(args.corpus, args.questions, split=args.split, configuration=args.config)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
        return 0
    except (OSError, sqlite3.Error, ValueError) as error:
        code = getattr(error, "code", "IO_ERROR" if isinstance(error, OSError) else
                       "STORE_ERROR" if isinstance(error, sqlite3.Error) else "INVALID_ARGUMENTS")
        # Exception payloads can contain document text or arbitrary filesystem contents.
        result = {"error": {"code": code, "message": str(error) if hasattr(error, "code")
                            else "Command failed; check arguments and local file access."}}
        if accepted is not None:
            result["documents"] = accepted
            if filename is not None:
                result["error"]["filename"] = Path(filename).name
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    sys.exit(main())
