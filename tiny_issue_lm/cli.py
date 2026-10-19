"""Command-line interface for training and generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import generate_text
from .training import TrainConfig, train_bigram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-issue-lm")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train the bigram model")
    train.add_argument("--text", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("artifacts/bigram.pt"))
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--context-length", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=0.05)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    generate = commands.add_parser("generate", help="sample from a checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        config = TrainConfig(
            steps=args.steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=args.device,
        )
        text = args.text.read_text(encoding="utf-8")
        result = train_bigram(text, args.output, config)
        print(f"saved checkpoint: {args.output}")
        print(result)
    else:
        print(
            generate_text(
                args.checkpoint,
                args.prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.device,
            )
        )


if __name__ == "__main__":
    main()

