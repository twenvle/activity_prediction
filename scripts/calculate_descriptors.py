#!/usr/bin/env python3
"""特徴量を実行するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parents[1] = parent.parent

from activity_prediction.descriptors import DescriptorsConfig, descriptors_csv


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(description="特徴量を取得します．")
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
    """CLI引数を解析して特徴量を取得する"""
    parser = build_parser()
    args = parser.parse_args(argv)

    config_kwargs = {
        "input_path": args.input,
        "output_path": args.output_dir / args.input.name,
    }

    if args.random_seed is not None:
        config_kwargs["random_seed"] = args.random_seed

    if args.encoding is not None:
        config_kwargs["encoding"] = args.encoding

    config = DescriptorsConfig(**config_kwargs)
    try:
        descriptors_csv(config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
