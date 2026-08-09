"""Fetch and verify public data used by the ICDE study."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FILES = {
    "yellow_tripdata_2025-01.parquet": (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet",
        "9af277e4c0d3f9deb30644da822981e1e7df6af58313170fd3aa8a474485488a",
    ),
    "taxi_zone_lookup.csv": (
        "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv",
        "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed",
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for filename, (url, expected) in FILES.items():
        target = RAW / filename
        if target.exists() and digest(target) == expected:
            print(f"verified {filename}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "QuaRM/0.1 research artifact"})
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        actual = digest(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual}")
        temporary.replace(target)
        print(f"downloaded and verified {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
