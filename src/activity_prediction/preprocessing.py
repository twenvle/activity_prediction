#!/usr/bin/env python3
"""CSV中のSMILESを検査し、学習に使用できる行だけを保存する前処理。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REQUIRED_COLUMNS = ("cid", "SMILES")

# 非金属，およびcam-b3lyp/6-311+G(d,p)で計算可能な原子
METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        31,
    }
    | set(range(21, 31))
    | set(range(37, 118))
)

EXCLUSION_REASONS = (
    "missing_smiles",
    "invalid_smiles",
    "contains_sulfonic_acid",
    "contains_phosphorus_oh",
    "not_benzoic_acid_derivative",
    "contains_metal",
    "ionic_smiles",
    "multiple_components",
    "3d_generation_failed",
    "missing_values",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """CSV前処理の入出力とSMILES検査条件。"""

    input_path: Path | str
    output_path: Path | str
    name_column: str = "cid"
    smiles_column: str = "SMILES"
    encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not str(self.name_column).strip():
            raise ValueError("name_columnを指定してください")
        if not str(self.smiles_column).strip():
            raise ValueError("smiles_columnを指定してください")
        if not 0 <= self.random_seed <= 2_147_483_646:
            raise ValueError("random_seedは0以上2147483646以下にしてください")

        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path))


@dataclass
class PreprocessingResult:
    """前処理後のDataFrameと除外件数。"""

    config: PreprocessingConfig
    dataframe: Any
    rows_read: int
    rows_written: int
    exclusion_counts: dict[str, int]

    @property
    def rows_excluded(self) -> int:
        return self.rows_read - self.rows_written


def _import_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "前処理にはpandasとRDKitが必要です。次のコマンドで導入してください:\n"
            "pip install pandas rdkit"
        ) from exc
    return {"pd": pd, "Chem": Chem, "AllChem": AllChem}


def _resolve_column(columns: Sequence[str], requested: str) -> str:
    """大文字・小文字を区別せず、要求された列を一意に解決する。"""
    matches = [
        column for column in columns if str(column).casefold() == requested.casefold()
    ]
    if not matches:
        raise ValueError(f"必須列'{requested}'が入力CSVにありません")
    if len(matches) > 1:
        raise ValueError(f"大文字・小文字だけが異なる重複列があります: {requested}")
    return str(matches[0])


def _normalize_smiles(value: Any, pd: Any) -> str | None:
    if pd.isna(value):
        return None
    smiles = str(value).strip()
    return smiles or None


def _phthalic_acid_derivatives(mol: Any, Chem: Any) -> bool:
    mol = Chem.AddHs(mol)
    phthalic_acid = Chem.MolFromSmarts(
        "[c:5]1([CX3:1](=[OX1:2])[OX2:3][H:4])[c:6]([CX3:11](=[OX1:12])[OX2:13][H:14])[c:7][c:8][c:9][c:10]1"
    )
    return mol.HasSubstructMatch(phthalic_acid)


def _excluded_acid_group_reason(mol: Any, Chem: Any) -> str | None:
    """スルホン酸基（-SO3H）またはリン原子に結合したP-OHを検出する。"""
    excluded_groups = (
        (
            "contains_sulfonic_acid",
            Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]"),
        ),
        ("contains_phosphorus_oh", Chem.MolFromSmarts("[P]-[OX2H1]")),
    )
    for reason, pattern in excluded_groups:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return reason
    return None


def _contains_metal(mol: Any) -> bool:
    return any(atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms())


def _is_ionic(mol: Any) -> bool:
    """正味電荷0の双性イオンも含め、形式電荷を持つ分子をイオンとする。"""
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return total_charge != 0


def _multiple_components(mol: Any, Chem: Any) -> bool:
    """分子が複数の化学種からなる場合はTrueを返す。"""
    return len(Chem.GetMolFrags(mol)) > 1


def _embedding_parameters(AllChem: Any, random_seed: int, use_random: bool) -> Any:
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = random_seed
    parameters.enforceChirality = True
    parameters.useRandomCoords = use_random
    return parameters


def _can_generate_3d(mol: Any, Chem: Any, AllChem: Any, random_seed: int) -> bool:
    """ETKDGで3D配座を作る。通常座標で失敗した場合はランダム座標で再試行する。"""
    seeds = (random_seed, random_seed + 1)
    for attempt, seed in enumerate(seeds):
        try:
            molecule_3d = Chem.AddHs(Chem.Mol(mol))
            parameters = _embedding_parameters(
                AllChem,
                seed,
                use_random=attempt == 1,
            )
            status = AllChem.EmbedMolecule(molecule_3d, parameters)
            if status == 0 and molecule_3d.GetNumConformers() > 0:
                return True
        except (RuntimeError, ValueError):
            continue
    return False


def _molecule_exclusion_reason(
    smiles_value: Any,
    *,
    pd: Any,
    Chem: Any,
    AllChem: Any,
    random_seed: int,
) -> str | None:
    smiles = _normalize_smiles(smiles_value, pd)
    if smiles is None:
        return "missing_smiles"

    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except (RuntimeError, ValueError):
        mol = None
    # ダミー原子（*）を含むSMILESも、実体のある構造ではないものとして除外する。
    if mol is None or mol.GetNumAtoms() == 0:
        return "invalid_smiles"
    excluded_acid_group_reason = _excluded_acid_group_reason(mol, Chem)
    if excluded_acid_group_reason is not None:
        return excluded_acid_group_reason
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return "invalid_smiles"
    if not _phthalic_acid_derivatives(mol, Chem):
        return "not_phthalic_acid_derivative"
    if _contains_metal(mol):
        return "contains_metal"
    if _is_ionic(mol):
        return "ionic_smiles"
    if _multiple_components(mol, Chem):
        return "multiple_components"
    if not _can_generate_3d(mol, Chem, AllChem, random_seed):
        return "3d_generation_failed"
    return None


def _replace_blank_strings_with_missing(frame: Any, pd: Any) -> Any:
    output = frame.copy()
    text_columns = output.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        output[column] = output[column].replace(r"^\s*$", pd.NA, regex=True)
    return output


def preprocess_dataframe(
    frame: Any,
    *,
    name_column: str = "cid",
    smiles_column: str = "SMILES",
    random_seed: int = 42,
    dependencies: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, int]]:
    """DataFrameを検査し、採用行と除外理由別件数を返す。"""
    deps = dict(dependencies or _import_dependencies())
    pd = deps["pd"]
    Chem = deps["Chem"]
    AllChem = deps["AllChem"]

    if frame.empty:
        raise ValueError("入力CSVにデータ行がありません")
    actual_name_column = _resolve_column(list(frame.columns), name_column)
    actual_smiles_column = _resolve_column(list(frame.columns), smiles_column)
    if actual_name_column == actual_smiles_column:
        raise ValueError("name列とsmiles列は別の列にしてください")

    actual_smiles_column = "SMILES"

    counts: Counter[str] = Counter({reason: 0 for reason in EXCLUSION_REASONS})
    kept_indices: list[Any] = []
    for index, smiles_value in frame[actual_smiles_column].items():
        reason = _molecule_exclusion_reason(
            smiles_value,
            pd=pd,
            Chem=Chem,
            AllChem=AllChem,
            random_seed=random_seed,
        )
        if reason is None:
            kept_indices.append(index)
        else:
            counts[reason] += 1

    structurally_valid = frame.loc[
        kept_indices,
        [actual_name_column, "Name", actual_smiles_column],
    ].copy()

    structurally_valid.columns = ["cid", "Name", "SMILES"]
    structurally_valid = _replace_blank_strings_with_missing(
        structurally_valid,
        pd,
    )

    complete_rows = structurally_valid.dropna(
        subset=["cid", "Name", "SMILES"],
    ).copy()

    counts["missing_values"] += len(structurally_valid) - len(complete_rows)
    complete_rows.reset_index(drop=True, inplace=True)
    return complete_rows, dict(counts)


def preprocess_csv(
    config: PreprocessingConfig | Mapping[str, Any],
) -> PreprocessingResult:
    """CSVを読み込み、条件を満たす完全な行だけを新しいCSVへ保存する。"""
    if not isinstance(config, PreprocessingConfig):
        config = PreprocessingConfig(**dict(config))

    deps = _import_dependencies()
    pd = deps["pd"]
    frame = pd.read_csv(config.input_path, encoding=config.encoding)
    cleaned, exclusion_counts = preprocess_dataframe(
        frame,
        name_column=config.name_column,
        smiles_column=config.smiles_column,
        random_seed=config.random_seed,
        dependencies=deps,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(
        config.output_path,
        index=False,
        encoding=config.output_encoding,
    )
    return PreprocessingResult(
        config=config,
        dataframe=cleaned,
        rows_read=len(frame),
        rows_written=len(cleaned),
        exclusion_counts=exclusion_counts,
    )
