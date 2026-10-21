#!/usr/bin/env python3
"""未評価化合物を予測するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parents[1] = parent.parent

from activity_prediction.model_prediction import PredictionConfig, predict_pka


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(description="未評価化合物を予測します．")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="raw CSV",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim",
        help="保存先フォルダ",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="乱数シード",
    )
    parser.add_argument(
        "--encoding",
        help="入力CSVの文字コード",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI引数を解析して未評価化合物を予測する"""
    parser = build_parser()
    args = parser.parse_args(argv)

    config_kwargs = {
        "input_path": args.input,
        "output_path": args.output_dir / args.input.name,
        "model": args.model,
    }

    if args.random_seed is not None:
        config_kwargs["random_seed"] = args.random_seed

    if args.encoding is not None:
        config_kwargs["encoding"] = args.encoding

    config = PredictionConfig(**config_kwargs)
    try:
        predict_pka(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
