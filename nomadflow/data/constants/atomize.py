import importlib.resources as impr
import pickle

from rdkit.Chem import AllChem
import numpy as np

import nomadflow.data.constants
from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Bond

FILE_DIR = impr.files(nomadflow.data.constants)

def _gen_atomized_standard_aa_dicts():
    with open(FILE_DIR / "standard_aas.pkl", 'rb') as fp:
        mol_data = pickle.load(fp)

    bond_data = {}
    for aa_name, ref_mol in mol_data.items():
        atom_order = const.ref_atoms[aa_name]

        # Remove hydrogens
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

        idx_map = {}  # Used for bonds later

        for i, atom in enumerate(ref_mol.GetAtoms()):
            # Get atom name, charge, element and reference coordinates
            atom_name = atom.GetProp("name")
            if atom_name != 'OXT':  # ignoring OXT to simplify things (and in most cases it's thrown away anyway)
                idx_map[i] = atom_order.index(atom_name)

        # Load bonds
        bonds = []
        unk_bond = const.bond_type_ids[const.unk_bond_type]
        for bond in ref_mol.GetBonds():
            idx_1 = bond.GetBeginAtomIdx()
            idx_2 = bond.GetEndAtomIdx()

            # Skip bonds with atoms ignored
            if (idx_1 not in idx_map) or (idx_2 not in idx_map):
                continue

            idx_1 = idx_map[idx_1]
            idx_2 = idx_map[idx_2]
            bond_type = bond.GetBondType().name
            bond_type = const.bond_type_ids.get(bond_type, unk_bond)
            bonds.append((idx_1, idx_2, bond_type))

        bond_data[aa_name] = bonds

    return bond_data

standard_aa_bonds = _gen_atomized_standard_aa_dicts()


def get_standard_protein_residue_bonds(res_name, atom_idx=0):
    # this is a slight abuse of data types, since we're actually making atom bonds
    bond_data = np.array(standard_aa_bonds[res_name], dtype=Bond)
    bond_data['atom_1'] += atom_idx
    bond_data['atom_2'] += atom_idx
    return bond_data