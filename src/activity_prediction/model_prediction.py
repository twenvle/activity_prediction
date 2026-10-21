#!/usr/bin/env python3
"""学習済みモデルによる候補予測をNotebookと外部スクリプトで共用する。

Notebook:
    config = PredictionConfig(method="ridge", summary_path="summary.json", model_path="model.pkl")
    result = ModelPredictor(config).predict(train_df, candidate_df, save_artifacts=False)
    result.predictions
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

METHODS = ("blr", "elasticnet", "gpr", "lasso", "ols", "pls", "ridge", "svr")
TARGETS = ("conversion", "selectivity", "yield")
AD_MODES = ("none", "hard", "soft")
AdMode = Literal["none", "hard", "soft"]
_SIGMA_FLOOR = 1e-12
_NUMERIC_RESULT_COLUMNS = (
    "mu",
    "sigma_raw",
    "sigma_cal",
    "ci_lower",
    "ci_upper",
    "current_best_y",
    "xi",
    "beta",
    "pi",
    "ei",
    "ucb",
    "ad_knn_distance",
    "ad_threshold",
    "ad_ratio",
    "ad_penalty",
    "ei_ad",
)
_PATH_FIELDS = (
    "summary_path",
    "model_path",
    "train_input_path",
    "candidate_input_path",
    "output_path",
    "output_dir",
)


@lru_cache(maxsize=1)
def _data_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Prediction requires numpy and pandas: pip install numpy pandas"
        ) from exc
    return np, pd


@lru_cache(maxsize=1)
def _normal_distribution() -> Any:
    try:
        from scipy.stats import norm
    except ImportError as exc:
        raise RuntimeError(
            "Acquisition functions require scipy: pip install scipy"
        ) from exc
    return norm


@lru_cache(maxsize=1)
def _nearest_neighbors() -> Any:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "AD calculation requires scikit-learn: pip install scikit-learn"
        ) from exc
    return NearestNeighbors


@lru_cache(maxsize=1)
def _chemistry() -> Any:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "SMILES comparison requires RDKit: pip install rdkit"
        ) from exc
    return Chem


@dataclass(frozen=True)
class PredictionConfig:
    """共通の予測条件。DataFrameやモデルを直接渡す場合は入力パスを省略可能。"""

    method: str
    summary_path: Path | str | None = None
    model_path: Path | str | None = None
    train_input_path: Path | str | None = None
    candidate_input_path: Path | str | None = None
    output_path: Path | str | None = None
    output_dir: Path | str | None = None
    target: str = "yield"
    xi: float = 0.5
    beta: float = 1.0
    ci_level: float = 0.95
    ad_k: int = 5
    ad_quantile: float = 0.95
    ad_mode: AdMode = "soft"
    sort_by: str = "ei_ad"
    encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.target not in TARGETS:
            raise ValueError(f"Unsupported target: {self.target}")
        if self.ad_mode not in AD_MODES:
            raise ValueError(f"Unsupported ad_mode: {self.ad_mode}")
        for name in ("xi", "beta", "ci_level", "ad_quantile"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number.")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        if not 0 < self.ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1 (exclusive).")
        if not 0 <= self.ad_quantile <= 1:
            raise ValueError("ad_quantile must be between 0 and 1.")
        if (
            isinstance(self.ad_k, bool)
            or not isinstance(self.ad_k, int)
            or self.ad_k < 1
        ):
            raise ValueError("ad_k must be a positive integer.")
        if not isinstance(self.sort_by, str) or not self.sort_by.strip():
            raise ValueError("sort_by must be a non-empty column name.")
        for name in _PATH_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())

    @property
    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        directory = self.output_dir
        if directory is None:
            directory = Path(__file__).resolve().parents[1] / "out"
        today = date.today().strftime("%Y%m%d")
        return directory / f"{today}_{self.method}_{self.target}_output.csv"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        base_dir: Path | str | None = None,
    ) -> PredictionConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Prediction configuration must be a mapping.")
        options = dict(values)
        unknown = set(options) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown prediction settings: {', '.join(sorted(unknown))}"
            )
        if not options.get("method"):
            raise ValueError("method is required.")
        base = Path.cwd() if base_dir is None else Path(base_dir)
        for name in _PATH_FIELDS:
            value = options.get(name)
            if value is not None:
                path = Path(value).expanduser()
                options[name] = (path if path.is_absolute() else base / path).resolve()
        return cls(**options)


@dataclass
class PredictionResult:
    """予測・獲得関数・ADを含む表と、実際に保存したCSVのパス。"""

    config: PredictionConfig
    predictions: Any
    pipeline: Any
    summary: Mapping[str, Any]
    selected_descriptors: tuple[str, ...]
    current_best_y: float
    output_path: Path | None = None


def load_prediction_config(
    config_path: Path | str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> PredictionConfig:
    """JSON/YAMLを読み込み、相対パスを設定ファイル基準で解決する。"""
    path = Path(config_path).expanduser().resolve()
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML configuration requires PyYAML: pip install PyYAML"
            ) from exc
        try:
            values = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {exc}") from exc
    else:
        values = json.loads(content)
    if not isinstance(values, dict):
        raise ValueError("The configuration must contain a mapping.")
    if overrides:
        values.update(overrides)
    return PredictionConfig.from_mapping(values, base_dir=path.parent)


def canonicalize_smiles(value: Any) -> str | None:
    _, pd = _data_dependencies()
    if pd.isna(value):
        return None
    chemistry = _chemistry()
    mol = chemistry.MolFromSmiles(str(value))
    return None if mol is None else chemistry.MolToSmiles(mol, canonical=True)


def calculate_ei(mu: Any, sigma: Any, current_best: float, xi: float) -> Any:
    np, _ = _data_dependencies()
    norm = _normal_distribution()
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    improvement = mu - float(current_best) - float(xi)
    ei = np.zeros_like(improvement, dtype=float)
    nonzero = sigma > _SIGMA_FLOOR
    z = improvement[nonzero] / sigma[nonzero]
    ei[nonzero] = improvement[nonzero] * norm.cdf(z) + sigma[nonzero] * norm.pdf(z)
    ei[~nonzero] = np.maximum(improvement[~nonzero], 0.0)
    return ei


def calculate_pi(mu: Any, sigma: Any, current_best: float, xi: float) -> Any:
    np, _ = _data_dependencies()
    norm = _normal_distribution()
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    improvement = mu - float(current_best) - float(xi)
    pi = np.zeros_like(improvement, dtype=float)
    nonzero = sigma > _SIGMA_FLOOR
    pi[nonzero] = norm.cdf(improvement[nonzero] / sigma[nonzero])
    pi[~nonzero] = (improvement[~nonzero] > 0).astype(float)
    return pi


def calculate_ucb(mu: Any, sigma: Any, beta: float) -> Any:
    np, _ = _data_dependencies()
    return np.asarray(mu, dtype=float) + float(beta) * np.asarray(sigma, dtype=float)


def compute_train_knn_distances(features: Any, k: int) -> Any:
    np, _ = _data_dependencies()
    n = features.shape[0]
    if n < 2:
        return np.zeros(n)
    effective_k = min(max(1, k), n - 1)
    neighbors = _nearest_neighbors()(n_neighbors=effective_k + 1, metric="euclidean")
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    return distances[:, 1 : effective_k + 1].mean(axis=1)


def compute_query_knn_distances(
    train_features: Any, query_features: Any, k: int
) -> Any:
    effective_k = min(max(1, k), train_features.shape[0])
    neighbors = _nearest_neighbors()(n_neighbors=effective_k, metric="euclidean")
    neighbors.fit(train_features)
    distances, _ = neighbors.kneighbors(query_features)
    return distances.mean(axis=1)


def compute_ad_penalty(ad_ratio: Any, mode: AdMode) -> Any:
    np, _ = _data_dependencies()
    ad_ratio = np.asarray(ad_ratio, dtype=float)
    if mode == "none":
        return np.ones_like(ad_ratio)
    if mode == "hard":
        return (ad_ratio <= 1.0).astype(float)
    if mode == "soft":
        return np.exp(-(ad_ratio**2))
    raise ValueError(f"Unsupported ad_mode: {mode}")


def _predict_mu_sigma(
    method: str,
    model: Any,
    query_features: Any,
    train_features: Any,
    y_train: Any,
) -> tuple[Any, Any]:
    np, _ = _data_dependencies()
    if method in {"gpr", "blr"}:
        mu, sigma_raw = model.predict(query_features, return_std=True)
    elif method in {"ols", "ridge", "lasso", "elasticnet", "pls", "svr"}:
        mu = np.asarray(model.predict(query_features)).ravel()
        train_predictions = np.asarray(model.predict(train_features)).ravel()
        rmse = float(np.sqrt(np.mean((y_train - train_predictions) ** 2)))
        sigma_raw = np.full(len(mu), max(rmse, _SIGMA_FLOOR))
    else:
        raise ValueError(f"Unsupported method: {method}")
    return (
        np.asarray(mu, dtype=float).ravel(),
        np.asarray(sigma_raw, dtype=float).ravel(),
    )


def transform_features(features: Any, scaler: Any | None) -> Any:
    return (
        features.to_numpy(dtype=float) if scaler is None else scaler.transform(features)
    )


def attach_known_flags(candidate_df: Any, train_df: Any) -> tuple[Any, Any]:
    _, pd = _data_dependencies()
    known_by_cas = pd.Series(False, index=candidate_df.index)
    known_by_smiles = pd.Series(False, index=candidate_df.index)
    if "cas" in candidate_df.columns and "cas" in train_df.columns:
        train_cas = set(train_df["cas"].dropna().astype(str))
        known_by_cas = candidate_df["cas"].astype(str).isin(train_cas)
    if "smiles" in candidate_df.columns and "smiles" in train_df.columns:
        train_smiles = set(train_df["smiles"].map(canonicalize_smiles).dropna())
        known_by_smiles = (
            candidate_df["smiles"].map(canonicalize_smiles).isin(train_smiles)
        )
    return known_by_cas, known_by_smiles


class ModelPredictor:
    """学習済みPipelineとsummaryを再利用して候補を予測する。

    pipeline/summaryを直接渡すとファイルからの読み込みを省略する。
    ファイルからのモデル読み込みは最初のpredict呼び出し時に行う。
    """

    def __init__(
        self,
        config: PredictionConfig | Mapping[str, Any],
        *,
        pipeline: Any | None = None,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, PredictionConfig)
            else PredictionConfig.from_mapping(config)
        )
        self.pipeline_ = pipeline
        self.summary_ = dict(summary) if summary is not None else None

    def _load_artifacts(self) -> tuple[Any, Mapping[str, Any], list[str], float]:
        config = self.config
        if self.pipeline_ is None:
            if config.model_path is None:
                raise ValueError("Provide pipeline or set model_path.")
            try:
                import joblib
            except ImportError as exc:
                raise RuntimeError(
                    "Model loading requires joblib: pip install joblib"
                ) from exc
            try:
                self.pipeline_ = joblib.load(config.model_path)
            except ImportError as exc:
                raise RuntimeError(
                    f"A package required by the saved model is missing: {exc}"
                ) from exc
        if self.summary_ is None:
            if config.summary_path is None:
                raise ValueError("Provide summary or set summary_path.")
            self.summary_ = json.loads(
                config.summary_path.read_text(encoding="utf-8-sig")
            )
        summary = self.summary_
        if not isinstance(summary, Mapping):
            raise ValueError("Model summary must contain a mapping.")
        steps = getattr(self.pipeline_, "named_steps", None)
        if not isinstance(steps, Mapping) or config.method not in steps:
            raise TypeError(f"Pipeline must contain the '{config.method}' step.")
        for name in ("method", "target"):
            if name in summary and summary[name] != getattr(config, name):
                raise ValueError(
                    f"Configuration {name} does not match the model summary."
                )
        descriptors = summary.get("final_best_combi_on_full_data")
        if not isinstance(descriptors, (list, tuple)) or not descriptors:
            raise ValueError("Summary must contain final_best_combi_on_full_data.")
        if any(not isinstance(item, str) or not item for item in descriptors):
            raise ValueError("Selected descriptors must be non-empty column names.")
        if len(set(descriptors)) != len(descriptors) or config.target in descriptors:
            raise ValueError(
                "Selected descriptors must be unique and exclude the target."
            )
        sigma_scale = 1.0
        if config.method == "gpr":
            details = summary.get("final_best_on_full_data", {})
            if not isinstance(details, Mapping):
                raise ValueError("final_best_on_full_data must contain a mapping.")
            sigma_scale = float(details.get("sigma_", 1.0))
            if not math.isfinite(sigma_scale) or sigma_scale < 0:
                raise ValueError("GPR sigma_ scale must be finite and non-negative.")
        return self.pipeline_, summary, list(descriptors), sigma_scale

    def predict(
        self,
        train_df: Any | None = None,
        candidate_df: Any | None = None,
        *,
        save_artifacts: bool = True,
    ) -> PredictionResult:
        """DataFrameまたは設定内のCSVを読み込み、候補の予測表を返す。"""
        config = self.config
        np, pd = _data_dependencies()
        if train_df is None:
            if config.train_input_path is None:
                raise ValueError("Provide train_df or set train_input_path.")
            train_df = pd.read_csv(config.train_input_path, encoding=config.encoding)
        if candidate_df is None:
            if config.candidate_input_path is None:
                raise ValueError("Provide candidate_df or set candidate_input_path.")
            candidate_df = pd.read_csv(
                config.candidate_input_path, encoding=config.encoding
            )
        pipeline, summary, descriptors, sigma_scale = self._load_artifacts()
        for frame, columns, name in (
            (train_df, descriptors + [config.target], "train_df"),
            (candidate_df, descriptors, "candidate_df"),
        ):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame.")
            if not frame.columns.is_unique:
                raise ValueError(f"{name} must not contain duplicate column names.")
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise KeyError(f"Missing columns in {name}: {missing}")
        train_df = train_df.dropna(subset=descriptors + [config.target]).reset_index(
            drop=True
        )
        candidate_df = candidate_df.reset_index(drop=True)
        if train_df.empty:
            raise ValueError("No training rows remain after dropna.")
        allowed_sort = (
            set(candidate_df.columns)
            | set(_NUMERIC_RESULT_COLUMNS)
            | {
                "in_ad",
                "known_any",
                "prediction_status",
            }
        )
        if config.sort_by not in allowed_sort:
            raise ValueError(f"Unknown sort_by column: {config.sort_by}")

        X_train = train_df[descriptors].astype(float)
        y_train = train_df[config.target].to_numpy(dtype=float)
        if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(y_train).all():
            raise ValueError("Training descriptors and target must be finite numbers.")
        current_best_y = float(y_train.max())
        output = candidate_df.copy()
        output["prediction_status"] = "missing_descriptor"
        for column in _NUMERIC_RESULT_COLUMNS:
            output[column] = np.nan
        output["in_ad"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
        known_cas, known_smiles = attach_known_flags(candidate_df, train_df)
        known_any = known_cas | known_smiles
        output["known_any"] = known_any
        valid_mask = candidate_df[descriptors].notna().all(axis=1)
        valid_idx = output.index[valid_mask]

        if len(valid_idx):
            scaler = pipeline.named_steps.get("scaler")
            X_query = candidate_df.loc[valid_mask, descriptors].astype(float)
            if not np.isfinite(X_query.to_numpy()).all():
                raise ValueError("Candidate descriptors must be finite numbers.")
            train_features = transform_features(X_train, scaler)
            query_features = transform_features(X_query, scaler)
            mu, sigma_raw = _predict_mu_sigma(
                config.method,
                pipeline.named_steps[config.method],
                query_features,
                train_features,
                y_train,
            )
            sigma_cal = sigma_raw * sigma_scale if config.method == "gpr" else sigma_raw
            train_knn_dist = compute_train_knn_distances(train_features, config.ad_k)
            ad_threshold = float(np.quantile(train_knn_dist, config.ad_quantile))
            query_knn_dist = compute_query_knn_distances(
                train_features, query_features, config.ad_k
            )
            ad_ratio = query_knn_dist / max(ad_threshold, _SIGMA_FLOOR)
            ad_penalty = compute_ad_penalty(ad_ratio, config.ad_mode)
            ei = calculate_ei(mu, sigma_cal, current_best_y, config.xi)
            pi = calculate_pi(mu, sigma_cal, current_best_y, config.xi)
            ucb = calculate_ucb(mu, sigma_cal, config.beta)
            ei_ad = ei * ad_penalty
            known = known_any.loc[valid_idx].to_numpy(dtype=bool)
            z = float(_normal_distribution().ppf(1.0 - (1.0 - config.ci_level) / 2.0))
            values = {
                "mu": mu,
                "sigma_raw": sigma_raw,
                "sigma_cal": sigma_cal,
                "ci_lower": mu - z * sigma_cal,
                "ci_upper": mu + z * sigma_cal,
                "current_best_y": current_best_y,
                "xi": config.xi,
                "beta": config.beta,
                "pi": np.where(known, 0.0, pi),
                "ei": np.where(known, 0.0, ei),
                "ucb": np.where(known, -np.inf, ucb),
                "ad_knn_distance": query_knn_dist,
                "ad_threshold": ad_threshold,
                "ad_ratio": ad_ratio,
                "ad_penalty": ad_penalty,
                "ei_ad": np.where(known, 0.0, ei_ad),
            }
            for column, value in values.items():
                output.loc[valid_idx, column] = value
            output.loc[valid_idx, "in_ad"] = ad_ratio <= 1.0
            output.loc[valid_idx, "prediction_status"] = "ok"

        by = [config.sort_by, "mu"] if config.sort_by != "mu" else ["mu"]
        output = output.sort_values(
            by=by, ascending=[False] * len(by), na_position="last"
        ).reset_index(drop=True)
        output_path = None
        if save_artifacts:
            output_path = config.resolved_output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False, encoding=config.encoding)
        return PredictionResult(
            config=config,
            predictions=output,
            pipeline=pipeline,
            summary=summary,
            selected_descriptors=tuple(descriptors),
            current_best_y=current_best_y,
            output_path=output_path,
        )


def predict_candidates(
    config: PredictionConfig | Mapping[str, Any],
    train_df: Any | None = None,
    candidate_df: Any | None = None,
    *,
    pipeline: Any | None = None,
    summary: Mapping[str, Any] | None = None,
    save_artifacts: bool = True,
) -> PredictionResult:
    """Notebook/外部スクリプト共通の関数。繰り返し予測する場合はModelPredictorを再利用する。"""
    return ModelPredictor(config, pipeline=pipeline, summary=summary).predict(
        train_df,
        candidate_df,
        save_artifacts=save_artifacts,
    )


def predict(
    train_df: Any,
    candidate_df: Any,
    summary_path: Path | str,
    joblib_path: Path | str,
    method: str,
    target: str = "yield",
    xi: float = 0.5,
    beta: float = 1.0,
    ci_level: float = 0.95,
    ad_k: int = 5,
    ad_quantile: float = 0.95,
    ad_mode: AdMode = "soft",
    sort_by: str = "ei_ad",
    *,
    output_path: Path | str | None = None,
    save_artifacts: bool = True,
) -> tuple[Any, Path | None]:
    """元のpredictと同じ引数・(DataFrame, Path)の戻り値を持つ互換関数。"""
    config = PredictionConfig(
        method=method,
        target=target,
        summary_path=summary_path,
        model_path=joblib_path,
        xi=xi,
        beta=beta,
        ci_level=ci_level,
        ad_k=ad_k,
        ad_quantile=ad_quantile,
        ad_mode=ad_mode,
        sort_by=sort_by,
        output_path=output_path,
    )
    result = predict_candidates(
        config, train_df, candidate_df, save_artifacts=save_artifacts
    )
    return result.predictions, result.output_path
