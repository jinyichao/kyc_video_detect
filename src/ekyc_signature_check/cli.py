import argparse
import json
import sys
from dataclasses import asdict

from .signature_complexity import analyze_signature_complexity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag a signature image as too simple (single letter, basic shape, or disconnected dots)."
    )
    parser.add_argument("image", help="Path to the signature image")
    parser.add_argument("--json", action="store_true", help="Print full result as JSON, including metrics")
    args = parser.parse_args()

    result = analyze_signature_complexity(args.image)

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        verdict = "TOO SIMPLE" if result.is_too_simple else "OK"
        print(f"Verdict: {verdict}")
        if result.reasons:
            print(f"  reasons: {', '.join(result.reasons)}")
        for k, v in result.metrics.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    sys.exit(0)


if __name__ == "__main__":
    main()
