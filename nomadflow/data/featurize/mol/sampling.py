from dataclasses import astuple
from typing import Tuple, Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol

import numpy as np

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import (
    Residue, Atom, Chain, Structure, Bond, SamplingResidue
)

from nomadflow.data.featurize.tokenize import convert_atom_str_to_tuple
from nomadflow.data.featurize.sampling import AtomData, ResidueData, ChainData


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to boltz2 format
    conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if len(conf_ids) > 0:
        conf_id = conf_ids[0]
        conformer = mol.GetConformer(conf_id)
        return conformer

    msg = "Conformer does not exist."
    raise ValueError(msg)


def smiles_to_struct(
    smiles: str,
    name: str = "LIG",
    chain_name: str = 'A',
    chain_idx: int = 0,
    noise_ligand: bool = True,
    trans_noise_std: int = 16,
) -> Structure:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    # Set atom names
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"{smiles} has an atom with a name longer than 4 characters: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    success = compute_3d_conformer(mol)
    if not success:
        msg = f"Failed to compute 3D conformer for {smiles}"
        raise ValueError(msg)

    mol_no_h = AllChem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol_no_h, cleanIt=True, force=True)

    residue_data = []
    atom_data = []
    bond_data = []
    res_idx = 0

    # Check if this is a single atom CCD residue
    if mol_no_h.GetNumAtoms() == 1:
        pos = np.array((0, 0, 0))
        mol_atom = mol_no_h.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(mol_atom.GetChiralTag()), unk_chirality
        )
        atom = AtomData(
            name=np.array(convert_atom_str_to_tuple(mol_atom.GetProp("name"))),
            element=mol_atom.GetAtomicNum(),
            charge=mol_atom.GetFormalCharge(),
            coords=pos,
            conformer=pos.copy(),
            is_present=True,
            chirality=chirality_type,
        )
        atom_data.append(astuple(atom))

        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ResidueData(
            name=name,
            res_type=unk_prot_id,
            res_idx=res_idx,
            atom_idx=0,
            atom_num=1,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
            is_copy=False
        )
        residue_data.append(astuple(residue))

    else:
        # Get reference conformer coordinates
        conformer = get_conformer(mol_no_h)

        # Parse each atom
        atom_idx = 0
        idx_map = {}  # Used for bonds later

        for i, mol_atom in enumerate(mol_no_h.GetAtoms()):
            # Get atom name, charge, element and reference coordinates
            atom_name = mol_atom.GetProp("name")
            charge = mol_atom.GetFormalCharge()
            element = mol_atom.GetAtomicNum()

            if noise_ligand:
                coords = np.random.randn(3) * trans_noise_std
            else:
                coords = conformer.GetAtomPosition(mol_atom.GetIdx())

            ref_coords = np.array((0.0, 0.0, 0.0))
            chirality_type = const.chirality_type_ids.get(
                str(mol_atom.GetChiralTag()), unk_chirality
            )

            # Add atom to list
            atom = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=True,
                chirality=chirality_type,
            )
            atom_data.append(astuple(atom))
            idx_map[i] = atom_idx
            atom_idx += 1  # noqa: SIM113

        # Load bonds
        unk_bond = const.bond_type_ids[const.unk_bond_type]
        for bond in mol_no_h.GetBonds():
            idx_1 = bond.GetBeginAtomIdx()
            idx_2 = bond.GetEndAtomIdx()

            # Skip bonds with atoms ignored
            if (idx_1 not in idx_map) or (idx_2 not in idx_map):
                continue

            idx_1 = idx_map[idx_1]
            idx_2 = idx_map[idx_2]
            start = min(idx_1, idx_2)
            end = max(idx_1, idx_2)
            bond_type = bond.GetBondType().name
            bond_type = const.bond_type_ids.get(bond_type, unk_bond)
            bond_data.append((start, end, bond_type))

        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ResidueData(
            name=name,
            res_type=unk_prot_id,
            res_idx=res_idx,
            atom_idx=0,
            atom_num=len(atom_data),
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
            is_copy=False,
        )
        residue_data.append(astuple(residue))

    chain_data = ChainData(
        name=chain_name,
        mol_type=const.chain_type_ids["PROTEIN"],
        entity_id=chain_idx,
        sym_id=chain_idx,
        asym_id=chain_idx,
        atom_idx=0,
        atom_num=len(atom_data),
        res_idx=res_idx,
        res_num=1,
        cyclic_period=0,
    )

    atoms = np.array(atom_data, dtype=Atom)
    bonds = np.array(bond_data, dtype=Bond)
    residues = np.array(residue_data, dtype=SamplingResidue)
    chains = np.array([astuple(chain_data)], dtype=Chain)

    struct = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.ones(chains.shape[0], dtype=bool),
    )
    return struct


def mol_to_struct(
    mol: Mol,
    name: str = "LIG",
    chain_name: str = 'A',
    chain_idx: int = 0,
    noise_ligand: bool = False,
    trans_noise_std: int = 16,
) -> Structure:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    mol = AllChem.AddHs(mol)
    smiles = Chem.MolToSmiles(mol)

    # Set atom names
    canonical_order = AllChem.CanonicalRankAtoms(mol)
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"{smiles} has an atom with a name longer than 4 characters: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    mol_no_h = AllChem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol_no_h, cleanIt=True, force=True)

    residue_data = []
    atom_data = []
    bond_data = []
    res_idx = 0

    # Check if this is a single atom CCD residue
    if mol_no_h.GetNumAtoms() == 1:
        pos = np.array((0, 0, 0))
        mol_atom = mol_no_h.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(mol_atom.GetChiralTag()), unk_chirality
        )
        atom = AtomData(
            name=np.array(convert_atom_str_to_tuple(mol_atom.GetProp("name"))),
            element=mol_atom.GetAtomicNum(),
            charge=mol_atom.GetFormalCharge(),
            coords=pos,
            conformer=pos.copy(),
            is_present=True,
            chirality=chirality_type,
        )
        atom_data.append(astuple(atom))

        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ResidueData(
            name=name,
            res_type=unk_prot_id,
            res_idx=res_idx,
            atom_idx=0,
            atom_num=1,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
            is_copy=False
        )
        residue_data.append(astuple(residue))

    else:
        # Get reference conformer coordinates
        conformer = get_conformer(mol_no_h)

        # Parse each atom
        atom_idx = 0
        idx_map = {}  # Used for bonds later

        for i, mol_atom in enumerate(mol_no_h.GetAtoms()):
            # Get atom name, charge, element and reference coordinates
            atom_name = mol_atom.GetProp("name")
            charge = mol_atom.GetFormalCharge()
            element = mol_atom.GetAtomicNum()

            if noise_ligand:
                coords = np.random.randn(3) * trans_noise_std
            else:
                coords = conformer.GetAtomPosition(mol_atom.GetIdx())

            ref_coords = np.array((0.0, 0.0, 0.0))
            chirality_type = const.chirality_type_ids.get(
                str(mol_atom.GetChiralTag()), unk_chirality
            )

            # Add atom to list
            atom = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=True,
                chirality=chirality_type,
            )
            atom_data.append(astuple(atom))
            idx_map[i] = atom_idx
            atom_idx += 1  # noqa: SIM113

        # Load bonds
        unk_bond = const.bond_type_ids[const.unk_bond_type]
        for bond in mol_no_h.GetBonds():
            idx_1 = bond.GetBeginAtomIdx()
            idx_2 = bond.GetEndAtomIdx()

            # Skip bonds with atoms ignored
            if (idx_1 not in idx_map) or (idx_2 not in idx_map):
                continue

            idx_1 = idx_map[idx_1]
            idx_2 = idx_map[idx_2]
            start = min(idx_1, idx_2)
            end = max(idx_1, idx_2)
            bond_type = bond.GetBondType().name
            bond_type = const.bond_type_ids.get(bond_type, unk_bond)
            bond_data.append((start, end, bond_type))

        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ResidueData(
            name=name,
            res_type=unk_prot_id,
            res_idx=res_idx,
            atom_idx=0,
            atom_num=len(atom_data),
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
            is_copy=False
        )
        residue_data.append(astuple(residue))

    chain_data = ChainData(
        name=chain_name,
        mol_type=const.chain_type_ids["PROTEIN"],
        entity_id=chain_idx,
        sym_id=chain_idx,
        asym_id=chain_idx,
        atom_idx=0,
        atom_num=len(atom_data),
        res_idx=res_idx,
        res_num=1,
        cyclic_period=0,
    )

    atoms = np.array(atom_data, dtype=Atom)
    bonds = np.array(bond_data, dtype=Bond)
    residues = np.array(residue_data, dtype=SamplingResidue)
    chains = np.array([astuple(chain_data)], dtype=Chain)

    struct = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=np.array([]),
        interfaces=np.array([]),
        mask=np.ones(chains.shape[0], dtype=bool),
    )
    return struct