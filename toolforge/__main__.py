"""Allow ``python -m toolforge`` as well as the ``toolforge`` entry point."""

from toolforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
