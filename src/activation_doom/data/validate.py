from __future__ import annotations

import argparse
import json
from pathlib import Path

from activation_doom.data.common import read_jsonl, validate_records, write_json


def validate_dataset(root: Path) -> dict:
    """Validate a collected dataset directory and write a validation report."""
    records = read_jsonl(root / "metadata.jsonl")
    config = json.loads((root / "dataset_config.json").read_text(encoding="utf-8"))
    expected = config.get("target_sample_count") if config.get("accepted_sample_count") == config.get("target_sample_count") else None
    report = validate_records(root, records, expected)
    write_json(root / "validation_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    """Build the dataset validation CLI parser."""
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    return p


def main() -> None:
    """Run dataset validation from command-line arguments."""
    validate_dataset(Path(parser().parse_args().dataset))


if __name__ == "__main__":
    main()
