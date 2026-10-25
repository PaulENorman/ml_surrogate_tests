"""Download one experiment's reports, checkpoint and fields from a Modal Volume."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import modal
from modal.volume import FileEntryType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--volume", default="drivaer-surface-ml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.run or not all(c.isalnum() or c in "_-" for c in args.run):
        parser.error("run must be a run directory name using letters, digits, underscores or dashes")
    root = PurePosixPath("runs") / args.run
    volume = modal.Volume.from_name(args.volume)
    entries = [e for e in volume.listdir(str(root), recursive=True) if e.type == FileEntryType.FILE]
    if not entries:
        raise ValueError("No artifacts found for this run")

    def download(entry):
        relative = PurePosixPath(entry.path.lstrip("/")).relative_to(root)
        target = args.output / relative
        if target.is_file() and target.stat().st_size == entry.size:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        with partial.open("wb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)
        if partial.stat().st_size != entry.size:
            raise ValueError(f"Download size mismatch for {relative}")
        partial.replace(target)
        return entry.size

    with ThreadPoolExecutor(max_workers=8) as pool:
        transferred = sum(pool.map(download, entries))
    print(f"Saved {len(entries)} artifact files ({transferred / 2**20:.1f} MiB transferred)")


if __name__ == "__main__":
    main()
