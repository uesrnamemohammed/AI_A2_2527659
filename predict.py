"""Command-line prediction: python predict.py --json samples/valid.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference import ModelsNotFoundError, predict_record  # noqa: E402
from src.validation import RecordValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one farm record stored as JSON.")
    parser.add_argument("--json", required=True, help="Path to a JSON file with one record.")
    args = parser.parse_args()
    try:
        record = json.loads(Path(args.json).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {args.json}"}, indent=2))
        return 2
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}, indent=2))
        return 2
    try:
        result = predict_record(record)
    except RecordValidationError as exc:
        print(json.dumps({"error": "invalid record", "details": exc.errors}, indent=2))
        return 2
    except ModelsNotFoundError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
