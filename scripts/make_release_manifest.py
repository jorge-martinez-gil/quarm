"""Hash the complete release surface after experiments and paper generation."""

from pathlib import Path

from quarm.io import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "results" / "release_manifest.json"


def main() -> int:
    included = []
    for pattern in [
        "pyproject.toml",
        "README.md",
        "ARTIFACT_EVALUATION.md",
        "src/quarm/*.py",
        "tests/*.py",
        "experiments/*.py",
        "scripts/*.py",
        "paper/*.md",
        "paper/*.bib",
        "artifacts/results/*.csv",
        "artifacts/results/manifest.json",
        "artifacts/figures/*.png",
        "output/pdf/*.pdf",
    ]:
        included.extend(ROOT.glob(pattern))
    files = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(set(included))
        if path.resolve() != OUTPUT.resolve()
    }
    write_json(
        OUTPUT,
        {
            "artifact": "QuaRM 0.1.0",
            "scope": "source, tests, evidence, figures, manuscript source, and final PDF",
            "algorithm": "SHA-256",
            "file_count": len(files),
            "files": files,
        },
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
