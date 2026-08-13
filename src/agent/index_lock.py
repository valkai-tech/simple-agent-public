"""Cross-process coordination for opening and rebuilding the search index."""

from pathlib import Path

from filelock import FileLock


def index_lock(index_dir: str) -> FileLock:
    """Return the lock shared by index builders and startup checks."""
    index_path = Path(index_dir).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(f"{index_path}.lock")
