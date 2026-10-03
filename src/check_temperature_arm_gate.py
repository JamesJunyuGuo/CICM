"""Check the Stage C Arm T identity gate before downstream jobs."""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    gate = summary.get("identity_gate", {})
    ok = bool(gate.get("pass")) and int(gate.get("n_mismatch", -1)) == 0
    print(
        json.dumps(
            {
                "summary": args.summary,
                "arm": summary.get("arm"),
                "model": summary.get("model"),
                "identity_gate": gate,
                "pass": ok,
            },
            indent=2,
        )
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
