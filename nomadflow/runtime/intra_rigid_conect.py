from __future__ import annotations

from pathlib import Path

from nomadflow.boltz.data import const


BACKBONE_GROUP_ATOMS = ("N", "CA", "C", "O", "CB")
PREFERRED_ALTLOCS = ("", "A")
PREFERRED_ALTLOC_RANK = {altloc: idx for idx, altloc in enumerate(PREFERRED_ALTLOCS)}
COARSE_GRAIN_SIDECHAIN_GROUPS = {
    "ALA": {2: [], 3: []},
    "ARG": {2: ["NE", "NH1", "NH2", "CZ"], 3: ["CB", "CG", "CD"]},
    "ASN": {2: ["CG", "ND2", "OD1"], 3: []},
    "ASP": {2: ["CG", "OD1", "OD2"], 3: []},
    "CYS": {2: ["CA", "CB", "SG"], 3: []},
    "GLN": {2: ["CG", "CD", "NE2", "OE1"], 3: []},
    "GLU": {2: ["CG", "CD", "OE1", "OE2"], 3: []},
    "GLY": {2: [], 3: []},
    "HIS": {2: ["CG", "CD2", "ND1", "CE1", "NE2"], 3: []},
    "ILE": {2: ["CB", "CG1", "CD1"], 3: ["CB", "CG1", "CG2"]},
    "LEU": {2: ["CG", "CD1", "CD2"], 3: []},
    "LYS": {2: ["CD", "CE", "NZ"], 3: ["CB", "CG", "CD"]},
    "MET": {2: ["CG", "SD", "CE"], 3: []},
    "PHE": {2: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"], 3: []},
    "PRO": {2: ["CB", "CG", "CD"], 3: []},
    "SER": {2: ["CA", "CB", "OG"], 3: []},
    "THR": {2: ["CB", "CG2", "OG1"], 3: []},
    "TRP": {2: ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"], 3: []},
    "TYR": {2: ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"], 3: []},
    "VAL": {2: ["CB", "CG1", "CG2"], 3: []},
}
STANDARD_RESIDUE_BONDS = {
    "ALA": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    "ARG": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")],
    "ASN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")],
    "ASP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")],
    "CYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "SG")],
    "GLN": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")],
    "GLU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")],
    "GLY": [("N", "CA"), ("CA", "C"), ("C", "O")],
    "HIS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2")],
    "ILE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")],
    "LEU": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")],
    "LYS": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
    "MET": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")],
    "PHE": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ")],
    "PRO": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")],
    "SER": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG")],
    "THR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")],
    "TRP": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2")],
    "TYR": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")],
    "VAL": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")],
}

def rigid_assignment_for_residue(res_name: str) -> dict[str, int]:
    rigid_map = {atom_name: 0 for atom_name in BACKBONE_GROUP_ATOMS}
    if res_name not in COARSE_GRAIN_SIDECHAIN_GROUPS:
        return rigid_map

    for atom_name in COARSE_GRAIN_SIDECHAIN_GROUPS[res_name][2]:
        rigid_map[atom_name] = 1
    for atom_name in COARSE_GRAIN_SIDECHAIN_GROUPS[res_name][3]:
        rigid_map[atom_name] = 2
    return rigid_map


def build_intra_rigid_conects(model_lines: list[str]) -> list[str]:
    # residue -> atom_name -> (serial, altloc_rank, res_name)
    residue_atoms: dict[tuple[str, str, str], dict[str, tuple[int, int, str]]] = {}
    fallback_rank = len(PREFERRED_ALTLOCS)

    for line in model_lines:
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        residue_key = (line[21].strip(), line[22:26].strip(), line[26].strip())
        atom_name = line[12:16].strip()
        alt_loc_rank = PREFERRED_ALTLOC_RANK.get(line[16].strip(), fallback_rank)
        atom_serial = int(line[6:11])
        res_name = line[17:20].strip()
        atoms_for_residue = residue_atoms.get(residue_key)
        if atoms_for_residue is None:
            residue_atoms[residue_key] = {atom_name: (atom_serial, alt_loc_rank, res_name)}
            continue
        current_atom = atoms_for_residue.get(atom_name)
        if current_atom is None:
            atoms_for_residue[atom_name] = (atom_serial, alt_loc_rank, res_name)
            continue
        if alt_loc_rank < current_atom[1]:
            atoms_for_residue[atom_name] = (atom_serial, alt_loc_rank, res_name)

    conect_pairs: set[tuple[int, int]] = set()
    for atoms_for_residue in residue_atoms.values():
        if not atoms_for_residue:
            continue

        res_name = next(iter(atoms_for_residue.values()))[2]
        if res_name not in const.ref_atoms or res_name not in COARSE_GRAIN_SIDECHAIN_GROUPS:
            continue

        rigid_map = rigid_assignment_for_residue(res_name)
        residue_bonds = STANDARD_RESIDUE_BONDS.get(res_name, [])

        for atom_1_name, atom_2_name in residue_bonds:
            atom_1_tuple = atoms_for_residue.get(atom_1_name)
            atom_2_tuple = atoms_for_residue.get(atom_2_name)
            if atom_1_tuple is None or atom_2_tuple is None:
                continue

            if rigid_map.get(atom_1_name, 0) != rigid_map.get(atom_2_name, 0):
                continue

            atom_1_serial = atom_1_tuple[0]
            atom_2_serial = atom_2_tuple[0]
            atom_pair = (
                min(atom_1_serial, atom_2_serial),
                max(atom_1_serial, atom_2_serial),
            )
            conect_pairs.add(atom_pair)

    return [f"CONECT{atom_1:>5}{atom_2:>5}" for atom_1, atom_2 in sorted(conect_pairs)]


def inject_conects_into_block(block_lines: list[str], end_record_prefix: str) -> tuple[list[str], int]:
    conect_lines = build_intra_rigid_conects(block_lines)
    output_lines = []
    inserted = False

    for line in block_lines:
        if line.startswith("CONECT"):
            continue
        if line.startswith(end_record_prefix):
            output_lines.extend(conect_lines)
            output_lines.append(line)
            inserted = True
        else:
            output_lines.append(line)

    if not inserted:
        output_lines.extend(conect_lines)

    return output_lines, len(conect_lines)


def process_pdb_lines(lines: list[str]) -> tuple[list[str], int, int]:
    output_lines: list[str] = []
    total_conects = 0
    model_count = 0
    has_models = False
    model_block: list[str] = []
    inside_model = False

    for line in lines:
        if line.startswith("MODEL"):
            has_models = True
            inside_model = True
            model_block = [line]
            continue

        if inside_model:
            if line.startswith("CONECT"):
                continue
            model_block.append(line)
            if line.startswith("ENDMDL"):
                patched_block, num_conects = inject_conects_into_block(model_block, "ENDMDL")
                output_lines.extend(patched_block)
                total_conects += num_conects
                model_count += 1
                inside_model = False
                model_block = []
            continue

        if not line.startswith("CONECT"):
            output_lines.append(line)

    if inside_model and model_block:
        patched_block, num_conects = inject_conects_into_block(model_block, "ENDMDL")
        output_lines.extend(patched_block)
        total_conects += num_conects
        model_count += 1

    if not has_models:
        patched_block, num_conects = inject_conects_into_block(output_lines, "END")
        return patched_block, num_conects, 1

    return output_lines, total_conects, model_count


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_intra_rigid_conect{input_path.suffix}")


def process_pdb_file(input_pdb: str | Path, output_pdb: str | Path | None = None) -> tuple[Path, int, int]:
    input_path = Path(input_pdb)
    output_path = Path(output_pdb) if output_pdb else default_output_path(input_path)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    output_lines, total_conects, model_count = process_pdb_lines(lines)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return output_path, total_conects, model_count
