"""Convenience entry point for recursively inspecting one or both data sources."""

try:
    from .generate_metadata import main
except ImportError:
    from generate_metadata import main


if __name__ == "__main__":
    raise SystemExit(main())