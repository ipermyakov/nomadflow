"""
From boltz
https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/write/pdb.py
"""
import re
from typing import Optional

from rdkit import Chem
from torch import Tensor

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Structure


def to_pdb(
    structure: Structure,
    plddts: Optional[Tensor] = None,
    rename_chains: bool = False
) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output PDB file

    """
    pdb_lines: list[str] = []
    append_line = pdb_lines.append

    atom_index = 1
    atom_reindex_ter: list[int] = []

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()
    element_symbol_cache: dict[int, str] = {}
    atom_name_cache: dict[tuple[int, ...], str] = {}
    plddt_values = None if plddts is None else plddts.detach().cpu().numpy()

    # Index into plddt tensor for current residue.
    res_num = 0
    # Tracks non-ligand plddt tensor indices,
    # Initializing to -1 handles case where ligand is resnum 0
    prev_polymer_resnum = -1
    # Tracks ligand indices.
    ligand_index_offset = 0

    CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if structure.chains.shape[0] > len(CHAIN_ALPHABET):
        raise ValueError((
            f"the input structure has {structure.chains.shape[0]} > {len(CHAIN_ALPHABET)} chains "
            "which is more than possible to represent using the PDB format"
        ))

    chain_tag_idx = 0
    chain_tag_mapping = {}

    # use the structure mask
    # so we have an easy way of controlling which chains are outputted
    chains = structure.chains[structure.mask]

    # Add all atom sites.
    for chain in chains:
        # We rename the chains in alphabetical order
        chain_idx = chain["asym_id"]

        if rename_chains:
            if chain["name"] not in chain_tag_mapping:
                chain_tag_mapping[chain["name"]] = CHAIN_ALPHABET[chain_tag_idx]
                chain_tag_idx += 1
            chain_tag = chain_tag_mapping[chain["name"]]
        else:
            chain_tag = chain["name"]
            assert len(chain_tag) == 1, f"chain tags must be single characters but we encountered chain tag {chain_tag}"

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        residues = structure.residues[res_start:res_end]
        for residue in residues:
            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            atoms = structure.atoms[atom_start:atom_end]
            atom_coords = atoms["coords"]

            record_type = (
                "ATOM"
                if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                else "HETATM"
            )

            for i, atom in enumerate(atoms):
                # This should not happen on predictions, but just in case.
                if not atom["is_present"]:
                    continue

                raw_name = tuple(int(c) for c in atom["name"] if c)
                name = atom_name_cache.get(raw_name)
                if name is None:
                    name = "".join(chr(c + 32) for c in raw_name)
                    atom_name_cache[raw_name] = name
                name = name if len(name) == 4 else f" {name}"  # noqa: PLR2004
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element_idx = int(atom["element"])
                element = element_symbol_cache.get(element_idx)
                if element is None:
                    element = periodic_table.GetElementSymbol(element_idx).upper()
                    element_symbol_cache[element_idx] = element
                charge = ""
                residue_index = residue["res_idx"] + 1
                pos = atom_coords[i]
                res_name_3 = (
                    "LIG" if record_type == "HETATM" else str(residue["name"][:3])
                )

                if record_type != 'HETATM':
                    # The current residue plddt is stored at the res_num index unless a ligand has previouly been added.
                    b_factor = (
                        100.00
                        if plddt_values is None
                        else float(plddt_values[res_num + ligand_index_offset]) * 100.0
                    )
                    prev_polymer_resnum = res_num
                else:
                    # If not a polymer resnum, we can get index into plddts by adding offset relative to previous polymer resnum.
                    ligand_index_offset += 1
                    b_factor = (
                        100.00
                        if plddt_values is None
                        else float(plddt_values[prev_polymer_resnum + ligand_index_offset]) * 100.0
                    )

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_index:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                append_line(atom_line)
                atom_reindex_ter.append(atom_index)
                atom_index += 1

            if record_type != 'HETATM':
                res_num += 1

        should_terminate = chain_idx < (len(structure.chains) - 1)
        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_name_3:>3} "
                f"{chain_tag:>1}{residue_index:>4}"
            )
            append_line(chain_termination_line)
            atom_index += 1

    # Dump CONECT records.
    for bonds in [structure.bonds, structure.connections]:
        for bond in bonds:
            atom1 = structure.atoms[bond["atom_1"]]
            atom2 = structure.atoms[bond["atom_2"]]
            if not atom1["is_present"] or not atom2["is_present"]:
                continue
            atom1_idx = atom_reindex_ter[bond["atom_1"]]
            atom2_idx = atom_reindex_ter[bond["atom_2"]]
            conect_line = f"CONECT{atom1_idx:>5}{atom2_idx:>5}"
            append_line(conect_line)

    append_line("END")
    append_line("")
    return "\n".join(line.ljust(80) for line in pdb_lines)