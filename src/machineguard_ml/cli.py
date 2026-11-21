from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_table, split_features_target
from .workflows import run_classification, run_regression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Equipment fault classification and indicator regression")
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name in ("classify", "regress"):
        command = subparsers.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--target", required=True)
        command.add_argument("--features", nargs="+")
        command.add_argument("--sheet")
        command.add_argument("--test-ratio", type=float, default=0.2)
        command.add_argument("--output-dir", type=Path, default=Path("output"))
        command.add_argument("--random-state", type=int, default=42)
    regression = subparsers.choices["regress"]
    regression.add_argument("--hidden-size", type=int, default=16)
    regression.add_argument("--epochs", type=int, default=500)
    regression.add_argument("--learning-rate", type=float, default=0.05)
    regression.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_table(args.input, sheet=args.sheet)
    features, target, selected = split_features_target(frame, args.target, args.features)
    if args.task == "classify":
        result = run_classification(features, target, args.test_ratio, args.random_state)
    else:
        result = run_regression(
            features,
            target,
            test_ratio=args.test_ratio,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    report = {"task": args.task, "features": selected, "target": args.target, **result.metrics}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
