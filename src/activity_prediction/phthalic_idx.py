from typing import Any


def detect_phtalicacid_idx(smiles: str, Chem: Any) -> list[list[int]]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Detect the indices of the atoms in the phthalic acid scaffold.
    carboxyl_pattern = Chem.MolFromSmarts(
        "[c:1]1([CX3:7](=[OX1:8])[OX2H1:9][H:13])[c:2]([CX3:10](=[OX1:11])[OX2H1:12][H:14])[c:3][c:4][c:5][c:6]1"
    )
    phthalic_matches = mol.GetSubstructMatches(carboxyl_pattern)
    if len(phthalic_matches) > 0:
        return phthalic_matches
    else:
        return None


def carboxyl_idx(smiles: str, Chem: Any) -> list[list[int]]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    carboxyl_sites = []
    phthalicacids = detect_phtalicacid_idx(smiles, Chem)
    for phthalicacid in phthalicacids:
        carboxyl_site = [
            phthalicacid[1],
            phthalicacid[2],
            phthalicacid[3],
            phthalicacid[4],
            phthalicacid[6],
            phthalicacid[7],
            phthalicacid[8],
            phthalicacid[9],
        ]
        carboxyl_sites.append(carboxyl_site)
    return carboxyl_sites


def hydrogen_idx(smiles: str, Chem: Any) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    hydrogen_sites = []
    phthalicacids = detect_phtalicacid_idx(smiles, Chem)
    for phthalicacid in phthalicacids:
        hydrogen_site1 = [phthalicacid[4]]
        hydrogen_site2 = [phthalicacid[9]]
        hydrogen_sites.append(hydrogen_site1)
        hydrogen_sites.append(hydrogen_site2)
    return hydrogen_sites


def oxygen_idx(smiles: str, Chem: Any) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    oxygen_sites = []
    phthalicacids = detect_phtalicacid_idx(smiles, Chem)
    for phthalicacid in phthalicacids:
        oxygen_site1 = [phthalicacid[3]]
        oxygen_site2 = [phthalicacid[8]]
        oxygen_sites.append(oxygen_site1)
        oxygen_sites.append(oxygen_site2)
    return oxygen_sites


def benzene_idx(smiles: str, Chem: Any) -> list[list[int]]:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    benzene_sites = []
    phthalicacids = detect_phtalicacid_idx(smiles, Chem)
    for phthalicacid in phthalicacids:
        benzene_site = [
            phthalicacid[10],
            phthalicacid[11],
            phthalicacid[12],
            phthalicacid[13],
        ]
        benzene_sites.append(benzene_site)
    return benzene_sites
