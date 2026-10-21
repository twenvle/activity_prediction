#!/usr/bin/env python3
"""量子化学ログとSMILESから記述子を抽出する共通モジュール。"""

from __future__ import annotations

import codecs
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ATOM_SYMBOLS = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}

DESCRIPTOR_COLUMNS = (
    "sasa",
    "polar",
    "h_nbo_charge",
    "o_nbo_charge",
    "molecular_volume_A3",
    "dipole_moment_debye",
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "omega",
    "3and6",
    "4and5",
    "logp",
    "hbd",
    "hba",
)

_ON_ERROR_VALUES = ("record", "raise")


def _same_file(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


@dataclass(frozen=True)
class DescriptorExtractionConfig:
    """記述子抽出の入力CSV、出力CSV、量子化学logdataの場所と処理条件。"""

    input_path: Path | str
    output_path: Path | str
    logdata_path: Path | str
    invalid_output_path: Path | str | None = None
    labeled: bool = True
    name_column: str = "name"
    smiles_column: str = "smiles"
    input_encoding: str = "utf-8-sig"
    output_encoding: str = "utf-8-sig"
    log_encoding: str = "utf-8"
    on_error: str = "record"

    def __post_init__(self) -> None:
        for name in ("name_column", "smiles_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if self.name_column == self.smiles_column:
            raise ValueError("name_column and smiles_column must be different.")
        for name in ("input_encoding", "output_encoding", "log_encoding"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty encoding name.")
            codecs.lookup(value)
        if self.on_error not in _ON_ERROR_VALUES:
            raise ValueError(f"on_error must be one of: {', '.join(_ON_ERROR_VALUES)}")
        if not isinstance(self.labeled, bool):
            raise TypeError("labeled must be a boolean.")
        for name in (
            "input_path",
            "output_path",
            "logdata_path",
            "invalid_output_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser())
        invalid_path = self.invalid_output_path
        if invalid_path is None:
            invalid_path = self.output_path.with_name(
                f"{self.output_path.stem}_invalid{self.output_path.suffix}"
            )
            object.__setattr__(self, "invalid_output_path", invalid_path)
        for first, second, message in (
            (self.input_path, self.output_path, "input_path and output_path"),
            (
                self.input_path,
                self.invalid_output_path,
                "input_path and invalid_output_path",
            ),
            (
                self.output_path,
                self.invalid_output_path,
                "output_path and invalid_output_path",
            ),
        ):
            if _same_file(first, second):
                raise ValueError(f"{message} must refer to different files.")


@dataclass
class DescriptorExtractionResult:
    """抽出できた行、失敗した行、処理件数、実際に保存したパス。"""

    config: DescriptorExtractionConfig
    dataframe: Any
    invalid_dataframe: Any
    rows_read: int
    rows_written: int
    error_counts: dict[str, int]

    @property
    def rows_invalid(self) -> int:
        return self.rows_read - self.rows_written


class DescriptorExtractionError(RuntimeError):
    """入力行にCAS、SMILES、logdata、計算上の問題がある場合の例外。"""


def _import_dependencies() -> dict[str, Any]:
    try:
        import cclib
        import freesasa
        import numpy as np
        import pandas as pd

        try:
            from . import phthalic_idx
        except ImportError:
            import phthalic_idx
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, Lipinski
    except ImportError as exc:
        raise RuntimeError(
            "Descriptor extraction requires cclib, freesasa, numpy, pandas and RDKit. "
            "Install the project requirements before running extraction."
        ) from exc
    return {
        "cclib": cclib,
        "freesasa": freesasa,
        "np": np,
        "pd": pd,
        "phthalic_idx": phthalic_idx,
        "Chem": Chem,
        "AllChem": AllChem,
        "Descriptors": Descriptors,
        "Lipinski": Lipinski,
    }


def find_phthalic_sites(
    smiles: str, dependencies: Mapping[str, Any]
) -> dict[str, tuple[int, ...]]:
    """フタル酸骨格の最初のSMARTS一致から用途別の0-based原子indexを返す。"""
    Chem = dependencies["Chem"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    matches = dependencies["phthalic_idx"].detect_phtalicacid_idx(smiles, Chem)
    if not matches:
        raise DescriptorExtractionError("Phthalic-acid scaffold was not found.")
    match = tuple(int(index) for index in matches[0])
    if len(match) < 14:
        raise DescriptorExtractionError(
            "Phthalic-acid SMARTS match has an unexpected shape."
        )
    return {
        "carboxyl_1": (match[0], match[1], match[2], match[3], match[4]),
        "carboxyl_2": (match[5], match[6], match[7], match[8], match[9]),
        "acid_hydrogens": (match[4], match[9]),
        "acid_oxygens": (match[2], match[7]),
        "benzene_positions": (match[10], match[11], match[12], match[13]),
    }


def calculate_frontier_orbitals(logdata: Any) -> dict[str, float]:
    homo_index = int(logdata.homos[0])
    homo_ev = float(logdata.moenergies[0][homo_index])
    lumo_ev = float(logdata.moenergies[0][homo_index + 1])
    gap_ev = lumo_ev - homo_ev
    if math.isclose(gap_ev, 0.0, abs_tol=1e-15):
        raise DescriptorExtractionError("HOMO-LUMO gap is zero.")
    return {
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": gap_ev,
        "omega": ((lumo_ev + homo_ev) / 2.0) ** 2 / gap_ev,
    }


def calculate_dipole_moment(logdata: Any, np: Any) -> dict[str, float]:
    return {"dipole_moment_debye": float(np.linalg.norm(logdata.moments[1]))}


def calculate_molecular_volume(
    logdata: Any, dependencies: Mapping[str, Any]
) -> dict[str, float]:
    lines = [str(len(logdata.atomnos)), ""]
    for atomic_number, coordinates in zip(logdata.atomnos, logdata.atomcoords[-1]):
        try:
            symbol = ATOM_SYMBOLS[int(atomic_number)]
        except KeyError as exc:
            raise DescriptorExtractionError(
                f"Unsupported atomic number for XYZ conversion: {atomic_number}"
            ) from exc
        x, y, z = (float(value) for value in coordinates)
        lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")
    mol = dependencies["Chem"].MolFromXYZBlock("\n".join(lines) + "\n")
    if mol is None:
        raise DescriptorExtractionError(
            "RDKit could not build a molecule from log geometry."
        )
    return {"molecular_volume_A3": float(dependencies["AllChem"].ComputeMolVolume(mol))}


def read_final_nbo_charges(logfile_path: Path, encoding: str = "utf-8") -> list[float]:
    """最後のNatural Population Analysis表から原子順の電荷を読む。"""
    lines = logfile_path.read_text(encoding=encoding, errors="replace").splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if "Summary of Natural Population Analysis" in line
    ]
    if not starts:
        raise DescriptorExtractionError("Natural Population Analysis was not found.")
    charges: list[float] = []
    for line in lines[starts[-1] + 1 :]:
        if charges and ("Total" in line or "====" in line):
            break
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            int(parts[1])
            charge = float(parts[2])
        except ValueError:
            continue
        charges.append(charge)
    if not charges:
        raise DescriptorExtractionError(
            "Natural Population Analysis contained no charges."
        )
    return charges


def calculate_nbo_descriptors(
    logfile_path: Path,
    hydrogen_sites: Sequence[int],
    oxygen_sites: Sequence[int],
    encoding: str = "utf-8",
) -> dict[str, float]:
    charges = read_final_nbo_charges(logfile_path, encoding)
    requested = tuple(hydrogen_sites) + tuple(oxygen_sites)
    if not requested or min(requested) < 0 or max(requested) >= len(charges):
        raise DescriptorExtractionError(
            f"NBO table has {len(charges)} atoms but requested indices are {requested}."
        )
    return {
        "h_nbo_charge": max(charges[index] for index in hydrogen_sites),
        "o_nbo_charge": min(charges[index] for index in oxygen_sites),
    }


def calculate_polarizability(logdata: Any, np: Any) -> dict[str, float]:
    return {"polar": float(np.trace(logdata.polarizabilities[-1]) / 3.0)}


def calculate_carboxyl_sasa(
    logdata: Any,
    carboxyl_groups: Sequence[Sequence[int]],
    dependencies: Mapping[str, Any],
) -> dict[str, float]:
    Chem = dependencies["Chem"]
    coordinates = [float(value) for row in logdata.atomcoords[-1] for value in row]
    periodic_table = Chem.GetPeriodicTable()
    radii = [periodic_table.GetRvdw(int(number)) for number in logdata.atomnos]
    result = dependencies["freesasa"].calcCoord(coordinates, radii)
    atom_areas = [
        float(result.atomArea(index)) for index in range(len(logdata.atomnos))
    ]
    group_areas = [
        sum(atom_areas[index] for index in group) for group in carboxyl_groups
    ]
    return {"sasa": float(sum(group_areas))}


def calculate_substitution_counts(
    smiles: str,
    benzene_positions: Sequence[int],
    Chem: Any,
) -> dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    mol = Chem.AddHs(mol)
    counts = {"3and6": 0, "4and5": 0}
    for position, atom_index in enumerate(benzene_positions):
        neighbors = {
            neighbor.GetSymbol()
            for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
        }
        if "H" not in neighbors:
            counts["3and6" if position in {0, 3} else "4and5"] += 1
    return counts


def calculate_rdkit_descriptors(
    smiles: str, dependencies: Mapping[str, Any]
) -> dict[str, float | int]:
    mol = dependencies["Chem"].MolFromSmiles(smiles)
    if mol is None:
        raise DescriptorExtractionError("SMILES could not be parsed by RDKit.")
    return {
        "logp": float(dependencies["Descriptors"].MolLogP(mol)),
        "hbd": int(dependencies["Lipinski"].NumHDonors(mol)),
        "hba": int(dependencies["Lipinski"].NumHAcceptors(mol)),
    }


def calculate_descriptors(
    smiles: str,
    logfile_path: Path,
    logdata: Any,
    sites: Mapping[str, Sequence[int]],
    dependencies: Mapping[str, Any],
    *,
    log_encoding: str = "utf-8",
) -> dict[str, Any]:
    """1分子についてすべての記述子を計算する。"""
    result: dict[str, Any] = {}
    result.update(
        calculate_carboxyl_sasa(
            logdata,
            (sites["carboxyl_1"], sites["carboxyl_2"]),
            dependencies,
        )
    )
    result.update(calculate_polarizability(logdata, dependencies["np"]))
    result.update(
        calculate_nbo_descriptors(
            logfile_path,
            sites["acid_hydrogens"],
            sites["acid_oxygens"],
            log_encoding,
        )
    )
    result.update(calculate_molecular_volume(logdata, dependencies))
    result.update(calculate_dipole_moment(logdata, dependencies["np"]))
    result.update(calculate_frontier_orbitals(logdata))
    result.update(
        calculate_substitution_counts(
            smiles,
            sites["benzene_positions"],
            dependencies["Chem"],
        )
    )
    result.update(calculate_rdkit_descriptors(smiles, dependencies))
    return result


def _find_logfile(name: str, config: DescriptorExtractionConfig) -> Path:
    if not name or any(character in name for character in "*?[]/\\"):
        raise DescriptorExtractionError(
            f"Invalid CAS value for a log filename: {name!r}"
        )
    pattern = (
        f"labeled/sub_{name}.log" if config.labeled else f"unlabeled/sub_{name}.log"
    )
    matches = sorted(
        path for path in config.logdata_path.glob(pattern) if path.is_file()
    )
    if not matches:
        raise DescriptorExtractionError(
            f"Log file was not found for {name}: {config.logdata_path / pattern}"
        )
    if len(matches) > 1:
        raise DescriptorExtractionError(
            f"Multiple log files were found for CAS {name}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _extract_row(
    row: Any,
    config: DescriptorExtractionConfig,
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    name = str(row[config.name_column]).strip()
    smiles = str(row[config.smiles_column]).strip()
    if not smiles:
        raise DescriptorExtractionError("SMILES is empty.")
    sites = find_phthalic_sites(smiles, dependencies)
    logfile_path = _find_logfile(name, config)
    logdata = dependencies["cclib"].io.ccread(str(logfile_path))
    if logdata is None:
        raise DescriptorExtractionError(
            f"cclib could not parse the log file: {logfile_path}"
        )
    return calculate_descriptors(
        smiles,
        logfile_path,
        logdata,
        sites,
        dependencies,
        log_encoding=config.log_encoding,
    )


def extract_descriptors(
    config: DescriptorExtractionConfig | Mapping[str, Any],
) -> DescriptorExtractionResult:
    """CSVから記述子を抽出し、成功行と失敗行をそれぞれ保存する。"""
    if not isinstance(config, DescriptorExtractionConfig):
        config = DescriptorExtractionConfig(**dict(config))

    dependencies = _import_dependencies()
    pd = dependencies["pd"]
    dataframe = pd.read_csv(config.input_path, encoding=config.input_encoding)
    if not dataframe.columns.is_unique:
        raise ValueError("Input data must not contain duplicate column names.")
    missing = [
        column
        for column in (config.name_column, config.smiles_column)
        if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    reserved_columns = set(DESCRIPTOR_COLUMNS) | {
        "source_index",
        "error_type",
        "error_message",
    }
    conflicts = sorted(reserved_columns.intersection(dataframe.columns))
    if conflicts:
        raise ValueError(
            "Input data already contains output columns: " + ", ".join(conflicts)
        )

    valid_rows: list[Any] = []
    feature_rows: list[dict[str, Any]] = []
    invalid_rows: list[Any] = []
    error_counts: dict[str, int] = {}
    for source_index, row in dataframe.iterrows():
        try:
            descriptors = _extract_row(row, config, dependencies)
        except Exception as exc:
            if config.on_error == "raise":
                raise DescriptorExtractionError(
                    f"Row {source_index} failed: {type(exc).__name__}: {exc}"
                ) from exc
            invalid = row.copy()
            invalid["source_index"] = source_index
            invalid["error_type"] = type(exc).__name__
            invalid["error_message"] = str(exc)
            invalid_rows.append(invalid)
            error_counts[type(exc).__name__] = (
                error_counts.get(type(exc).__name__, 0) + 1
            )
            continue
        valid = row.copy()
        valid["source_index"] = source_index
        valid_rows.append(valid)
        feature_rows.append(descriptors)

    source_columns = list(dataframe.columns) + ["source_index"]
    valid_frame = pd.DataFrame(valid_rows, columns=source_columns).reset_index(
        drop=True
    )
    feature_frame = pd.DataFrame(feature_rows, columns=DESCRIPTOR_COLUMNS)
    output = pd.concat([valid_frame, feature_frame], axis=1)
    invalid_columns = list(dataframe.columns) + [
        "source_index",
        "error_type",
        "error_message",
    ]
    invalid_output = pd.DataFrame(invalid_rows, columns=invalid_columns).reset_index(
        drop=True
    )

    invalid_output_path = config.invalid_output_path
    if invalid_output_path is None:  # __post_init__で設定されるため通常は到達しない
        raise RuntimeError("invalid_output_path was not initialized.")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_path, index=False, encoding=config.output_encoding)
    invalid_output.to_csv(
        invalid_output_path,
        index=False,
        encoding=config.output_encoding,
    )

    return DescriptorExtractionResult(
        config=config,
        dataframe=output,
        invalid_dataframe=invalid_output,
        rows_read=len(dataframe),
        rows_written=len(output),
        error_counts=error_counts,
    )
