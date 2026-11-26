"""Run the publication build CLI with ``python -m dubanevics2026topological.build``."""

from .cli import main

# The guard keeps spawned worker processes (Topographer's bootstrap pool) from
# re-running the CLI when they re-import this module.
if __name__ == "__main__":
    raise SystemExit(main())
