"""Report sensitive local files in the workspace without deleting anything."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SENSITIVE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".zip", ".pth", ".pt", ".ckpt", ".onnx"
}
SOURCE_EXTENSIONS = {".py"}
ABSOLUTE_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/]")


def git_paths(root: Path, *arguments: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *arguments],
        check=True,
        capture_output=True,
    )
    return {item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\0") if item}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / ".git").exists():
        parser.error(f"Not a Git worktree: {root}")

    tracked = git_paths(root, "--cached")
    other = git_paths(root, "--others", "--ignored", "--exclude-standard")
    workspace_files = sorted(tracked | other)
    sensitive: list[tuple[str, bool]] = []
    hardcoded_paths: list[tuple[str, int]] = []

    for relative in workspace_files:
        path = root / relative
        if not path.is_file():
            continue
        if path.suffix.casefold() in SENSITIVE_EXTENSIONS:
            sensitive.append((relative, relative in tracked))
        if path.suffix.casefold() in SOURCE_EXTENSIONS:
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if ABSOLUTE_WINDOWS_PATH.search(line):
                        hardcoded_paths.append((relative, line_number))
            except UnicodeDecodeError:
                continue

    tracked_sensitive = [path for path, is_tracked in sensitive if is_tracked]
    if tracked_sensitive:
        print("ERROR: sensitive image/archive/model files are Git-tracked:")
        for path in tracked_sensitive:
            print(f"  {path}")
    untracked_sensitive = [path for path, is_tracked in sensitive if not is_tracked]
    if untracked_sensitive:
        print("WARNING: sensitive files exist locally but are not Git-tracked:")
        for path in untracked_sensitive:
            print(f"  {path}")
        print("Keep them outside the repository where practical; this checker did not alter them.")
    if hardcoded_paths:
        print("WARNING: absolute Windows paths found in Python source:")
        for path, line_number in hardcoded_paths:
            print(f"  {path}:{line_number}")
    if not tracked_sensitive and not untracked_sensitive and not hardcoded_paths:
        print("Git safety check passed: no sensitive files or hardcoded Windows paths found.")
    elif not tracked_sensitive:
        print("No sensitive image/archive/model files are tracked by Git.")
    return 1 if tracked_sensitive or hardcoded_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())