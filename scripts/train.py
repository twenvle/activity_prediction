#!/usr/bin/env python3
"""設定ファイルを使って回帰モデルを学習するCLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # parents[1] = parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "model_training.yaml"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from activity_prediction.model_training import MODEL_NAMES  # noqa: E402
from activity_prediction.model_training import (
    TrainingResult,
    load_training_config,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    """コマンドライン引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="設定ファイルを使用してNested CVで回帰モデルを学習します。"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML設定ファイル（既定: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=MODEL_NAMES,
        help="学習モデル（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="特徴量CSV（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "-t",
        "--target-column",
        help="予測対象となる目的変数の列名（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="保存先フォルダ（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="外側K-Foldの分割数（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--inner-splits",
        type=int,
        help="内側K-Foldの分割数（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="乱数シード（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--encoding",
        help="入力CSVの文字コード（設定ファイルの値を上書き）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="成果物を保存せず、学習と評価だけを実行",
    )
    return parser


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """指定されたCLI引数だけを設定上書き用の辞書にする。"""
    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.input is not None:
        # CLIで指定した相対パスは、実行時のカレントディレクトリ基準にする。
        overrides["input_path"] = str(args.input.resolve())
    if args.target_column is not None:
        overrides["target_column"] = args.target_column
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir.resolve())
    if args.n_splits is not None:
        overrides["n_splits"] = args.n_splits
    if args.inner_splits is not None:
        overrides["inner_splits"] = args.inner_splits
    if args.random_state is not None:
        overrides["random_state"] = args.random_state
    if args.encoding is not None:
        overrides["encoding"] = args.encoding
    return overrides


def _print_training_result(result: TrainingResult) -> None:
    """CLI利用者向けに保存先と交差検証指標を表示する。"""
    if result.model_path is not None:
        print(f"モデル: {result.model_path}")
        print(f"評価指標: {result.metrics_path}")
        print(f"予測値: {result.predictions_path}")
    else:
        print("成果物: 保存していません")

    metrics = result.metrics
    print(
        "最良パラメータ: "
        + json.dumps(
            metrics["hyperparameter_search"]["final_best_parameters"],
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(
        f"Nested {result.config.n_splits}-Fold CV評価 "
        f"(inner={result.config.inner_splits}): "
        f"R2={metrics['test_metrics']['r2']:.6f}, "
        f"RMSE={metrics['test_metrics']['rmse']:.6f}, "
        f"MAE={metrics['test_metrics']['mae']:.6f} "
        f"(std: R2={metrics['test_metrics_std']['r2']:.6f}, "
        f"RMSE={metrics['test_metrics_std']['rmse']:.6f}, "
        f"MAE={metrics['test_metrics_std']['mae']:.6f})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI引数と設定ファイルを統合して学習を実行する。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_training_config(
            args.config,
            overrides=_build_overrides(args),
        )
        result = train_model(config, save_artifacts=not args.no_save)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    _print_training_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
