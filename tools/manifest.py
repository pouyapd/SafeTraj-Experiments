"""
SHA-256 manifest of the published thesis artifacts.

Text files are hashed with LF line endings (the bytes stored in git), so the
check gives the same result on Windows checkouts that convert to CRLF.

    python tools/manifest.py --check          # verify (exit code 1 on any difference)
    python tools/manifest.py --write          # (re)create docs/artifacts.sha256 — only for new artifacts
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "artifacts.sha256"
PATTERNS = ["results/**/*", "docs/figures/*.png", "notebooks/*.ipynb", "src/*.py"]
TEXT_SUFFIXES = {".csv", ".txt", ".py", ".ipynb", ".md", ".json"}


def file_hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def protected_files():
    files = set()
    for pattern in PATTERNS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def read_manifest():
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split("  ", 1)
            entries[rel] = digest
    return entries


def check():
    """Returns (missing, changed, unlisted) relative paths."""
    expected = read_manifest()
    actual = {p.relative_to(ROOT).as_posix(): p for p in protected_files()}
    missing = sorted(set(expected) - set(actual))
    changed = sorted(rel for rel, d in expected.items() if rel in actual and file_hash(actual[rel]) != d)
    unlisted = sorted(set(actual) - set(expected))
    return missing, changed, unlisted


def write():
    lines = ["# SHA-256 of the published thesis artifacts (text files hashed with LF line endings)."]
    for p in protected_files():
        lines.append(f"{file_hash(p)}  {p.relative_to(ROOT).as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines) - 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    args = p.parse_args()
    if args.write:
        print(f"wrote {write()} entries to {MANIFEST.relative_to(ROOT)}")
        return 0
    missing, changed, unlisted = check()
    for label, items in (("missing", missing), ("changed", changed), ("not in manifest", unlisted)):
        for rel in items:
            print(f"{label}: {rel}")
    ok = not (missing or changed or unlisted)
    print("all artifacts match the manifest" if ok else "ARTIFACTS DIFFER FROM THE MANIFEST")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
