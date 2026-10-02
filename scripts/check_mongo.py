"""Confirm Atlas Local accepts a connection and is a replica set.

Why directConnection is required: the container advertises hostname `mongodb`,
which does not resolve on the host. The node is still a replica set member,
which the checkpointer needs.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("MONGODB_URI is not set. Copy .env.example to .env.", file=sys.stderr)
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        ping = client.admin.command("ping")
        hello = client.admin.command("hello")
    except PyMongoError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    set_name = hello.get("setName")
    print(f"ping ok: {ping.get('ok')}")
    print(f"replica set: {set_name}")
    if not set_name:
        print("Server is not a replica set member.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
