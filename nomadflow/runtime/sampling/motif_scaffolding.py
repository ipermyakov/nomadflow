""" Code for generating motif scaffolding inputs

Conditions are specified with RFDiffusion-inspired strings which use a comma-separated list
to specify conditions along the sequence dimension.
- motif input can take the form ``X##'' to specify a single residue of chain ``X''
  or ``X##-##'' to specify a segment from chain ``X''
- sections of protein to design from scratch can be specified as ``##'' for a fixed number of residues
  or ``##-##'' for a variable range of possible residue spacings. If a variable range is specified,
  the number of residues for that segment will be sampled uniformly at random from the range (inclusive)

The precise form of motif scaffolding will be inferred from the data provided in the input PDB file. More concretely,
- if the motif does not contain the necessary backbone atoms, the backbone frame will be designed from scratch
- if the motif does not contain the necessary sidechain atoms, the missing sidechain frame(s) will be designed from scratch
The residue's identity will also be read from the PDB file and used as conditioning input.
If you wish to redesign the sequence identity as well, modify the input PDB residue identity to be ``UNK''.

"""
import copy
import string
import functools as fn
from dataclasses import astuple
import re
from collections.abc import Sequence

from rdkit import Chem

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure as BPStructure

from scipy.spatial.transform import Rotation
import torch
import torch.nn.functional as F
import tree
from torch_geometric.data import HeteroData
import numpy as np

from nomadflow.boltz.data.types import Structure, Atom, Residue, Chain, SamplingResidue
from nomadflow.boltz.data import const

from nomadflow.openfold.data import residue_constants
from nomadflow.data.constants import coarse_grain as cg
from nomadflow.openfold.data import data_transforms
from nomadflow.openfold.utils import rigid_utils as ru
from nomadflow.data.featurize.tokenize import Tokenized, convert_atom_str_to_tuple
from nomadflow.data.featurize.sampling import generate_protein_structure_template, sample_noise_from_struct_template, ResidueData, AtomData, ChainData
from nomadflow.data.featurize.assembler import featurize_inference
from .benchmark_metadata import (
    build_motif_scaffolding_sample_metadata,
    build_motif_scaffolding_task_metadata,
)
from .task import SamplingTask


def dict_cat(data):
    return tree.map_structure(lambda *x: torch.cat(x), *data)


def residue_to_atom37(residue):
    residue_37 = torch.full((37, 3), torch.nan)
    aa_3lt = residue.get_resname()
    atoms = list(residue.get_atoms())
    for atom in atoms:
        atom_name = atom.get_name()
        # we ignore hydrogens
        if str(atom.element).upper() == 'H':
            # print("skipping H")
            continue
        atom_idx = residue_constants.atom_types.index(atom_name)
        residue_37[atom_idx] = torch.as_tensor(atom.get_coord())
    return residue_37, residue_constants.resname_to_idx[aa_3lt]


def biopython_to_boltz(residue, res_idx, atom_idx, noise_bb=True, noise_tip=True, noise_sidechain=True):
    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    res_name = residue.get_resname()
    res_ref_atoms = const.ref_atoms[res_name]
    ordered_atom_list = []

    atom_noising_mask = []

    for atom_name in res_ref_atoms:
        if atom_name in residue:
            atom = residue[atom_name]
            element_idx = periodic_table.GetAtomicNumber(atom.element)
            atom_data = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element_idx,
                charge=0,  # TODO: probs should get this from a reference
                coords=np.array(atom.coord),
                conformer=np.array((0.0, 0.0, 0.0)),  # not used by nomadflow
                is_present=atom.is_disordered() == 0,
                chirality=0  # TODO: probs should get this from a reference
            )
            ordered_atom_list.append(
                astuple(atom_data)
            )
            if atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                atom_noising_mask.append(noise_bb)
            else:
                if atom_name in cg.coarse_grain_sidechain_groups[res_name][2]:
                    atom_noising_mask.append(noise_sidechain and noise_tip)
                else:
                    atom_noising_mask.append(noise_sidechain)
        else:
            element_idx = periodic_table.GetAtomicNumber(atom_name[0])
            atom_data = AtomData(
                name=np.array(convert_atom_str_to_tuple(atom_name)),
                element=element_idx,
                charge=0, # TODO: probs should get this from a reference
                coords=np.array((0.0, 0.0, 0.0)),
                conformer=np.array((0.0, 0.0, 0.0)),  # not used by nomadflow
                is_present=False,
                chirality=0  # TODO: probs should get this from a reference
            )
            ordered_atom_list.append(
                astuple(atom_data)
            )
            atom_noising_mask.append(True)

    res_data = ResidueData(
        name=res_name,
        res_type=const.token_ids[res_name],
        res_idx=res_idx,
        atom_idx=atom_idx,
        atom_num=len(res_ref_atoms),
        atom_center=1,
        atom_disto=1,
        is_standard=True,
        is_present=True,
        is_copy=True
    )
    new_atom_idx = atom_idx + len(res_ref_atoms)
    return astuple(res_data), ordered_atom_list, atom_noising_mask, new_atom_idx


# TODO: idx strings dont take into account chain
# and automatically assigned to sample_chain_name
class MotifScaffoldingTask(SamplingTask):
    chain_alphabet: str = string.ascii_uppercase
    task_name: str = "motif_scaffolding"

    def __init__(
        self,
        pdb_contigs,
        pdb,
        num_samples,
        total_length,
        contigs_idx="",
        sample_contigs_idx_config=None,
        sample_chain_name='A',
        cg_version=1,
        redesign_contigs=None,
        multi_motif_se3=False,
        multi_motif_partition_mode="fragments",
        multi_motif_max_groups=8,
        multi_motif_se3_trans_std=16.0,
        motif_t_lock: float | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if kwargs.get("stage2_input_pdb"):
            pdb = kwargs["stage2_input_pdb"]
        if kwargs.get("stage2_pdb_contigs"):
            pdb_contigs = kwargs["stage2_pdb_contigs"]
        if kwargs.get("stage2_sample_chain_name"):
            sample_chain_name = kwargs["stage2_sample_chain_name"]

        self.native_pdb = pdb
        self.sample_chain_name = sample_chain_name
        self.generator = np.random.default_rng()
        # Motif benchmark PDBs often contain discontinuous-chain warnings from Biopython.
        # They are noisy but expected here, so keep parsing quiet.
        parser = PDBParser(QUIET=True)

        # a little roundabout but this makes mypy happy
        structure = parser.get_structure("", pdb)
        assert isinstance(structure, BPStructure)
        self.structure: BPStructure = structure


        if len(contigs_idx) > 0:
            assert sample_contigs_idx_config is None, "sample_contigs_idx_config and contigs_idx are mutually exclusive"
        if sample_contigs_idx_config is not None:
            assert len(contigs_idx) == 0, "sample_contigs_idx_config and contigs_idx are mutually exclusive"

        self.contigs_idx = contigs_idx
        self.pdb_contigs = pdb_contigs
        self.sample_contigs_idx_config = sample_contigs_idx_config

        def get_num_motif_res():
            contigs = [c.strip() for c in pdb_contigs.split(",")]
            num_res = 0

            for contig in contigs:
                if contig[0] in self.chain_alphabet:
                    if "-" in contig:
                        motif_start, motif_end = [int(i) for i in contig[1:].split("-")]
                        num_res += (motif_end - motif_start + 1)
                    else:
                        num_res += 1
            return num_res
        self.num_motif_res = get_num_motif_res()

        self.redesign_contigs = {}
        if redesign_contigs is not None and len(redesign_contigs) > 1:
            redesign_contigs = [c.strip() for c in redesign_contigs.split(",")]
            for contig in redesign_contigs:
                chain = contig[0]
                if chain not in self.redesign_contigs:
                    self.redesign_contigs[chain] = set()
                if "-" in contig:
                    start, end = [int(i) for i in contig[1:].split("-")]
                    self.redesign_contigs[chain].update(set(range(start, end+1)))
                else:
                    resid = int(contig[1:])
                    self.redesign_contigs[chain].add(resid)

        self.num_samples = num_samples
        self.cg_version = cg_version
        self.multi_motif_se3 = bool(multi_motif_se3)
        self.multi_motif_partition_mode = str(multi_motif_partition_mode)
        self.multi_motif_max_groups = int(multi_motif_max_groups)
        self.multi_motif_se3_trans_std = float(multi_motif_se3_trans_std)
        self.motif_t_lock = float(motif_t_lock) if motif_t_lock is not None else None

        self.total_len = [int(i) for i in total_length.split("-")]
        assert len(self.total_len) in (1, 2)

        cg_group_mask = [
            cg.cg_group_mask[residue_constants.restype_1to3[resname]]
            for resname in residue_constants.restypes
        ]
        cg_group_mask.append([0.0] * 4)
        self.cg_group_mask = torch.as_tensor(cg_group_mask, dtype=torch.bool)[:, (0, 2, 3)]


    def _sample_idx_str(self, contigs_str):
        contigs = [c.strip() for c in contigs_str.split(",")]
        idx = []

        current_idx = 0

        for contig in contigs:
            if contig[0] in self.chain_alphabet:
                chain = contig[0]
                if "-" in contig:
                    motif_start, motif_end = [int(i) for i in contig[1:].split("-")]
                    for _ in range(motif_start, motif_end+1):
                        idx.append(current_idx)
                        current_idx += 1
                else:
                    motif_resid = int(contig[1:])
                    idx.append(current_idx)
                    current_idx += 1
            else:
                if "-" in contig:
                    lower, upper = [int(i) for i in contig.split("-")]
                    length = np.random.randint(lower, upper+1)
                    current_idx += length
                else:
                    length = int(contig)
                    current_idx += length

        return ",".join([str(i) for i in idx])

    def _parse_idx_str(self, idx_str, pdb_contigs_str):
        if len(idx_str) == 0:
            # blank str means all unindexed
            idxs = []
            pdb_contigs = [c.strip() for c in pdb_contigs_str.split(",")]
            for contig in pdb_contigs:
                if contig[0] in self.chain_alphabet:
                    if "-" in contig:
                        motif_start, motif_end = [int(i) for i in contig[1:].split("-")]
                        for _ in range(motif_start, motif_end + 1):
                            idxs.append(-1)
                    else:
                        idxs.append(-1)
                else:
                    raise ValueError()
        else:
            idx_contigs = [c.strip() for c in idx_str.split(",")]
            idxs = []

            for contig in idx_contigs:
                if contig.startswith("[") and contig.endswith("]"):
                    # unindexed residue
                    for _ in range(int(contig[1:-1])):
                        idxs.append(-1)
                elif "-" in contig:
                    # index range
                    motif_start, motif_end = [int(i) for i in contig.split("-")]
                    for i in range(motif_start, motif_end + 1):
                        idxs.append(i)
                else:
                    # there might be some adversarial case where this check holds but this should be fine for now
                    assert str(int(contig)) == contig
                    motif_resid = int(contig)
                    idxs.append(motif_resid)

        motif_seq_idx = torch.as_tensor(idxs, dtype=torch.long)
        motif_res_is_unindexed = (motif_seq_idx == -1)
        motif_seq_idx[motif_res_is_unindexed] = 0

        return motif_seq_idx, motif_res_is_unindexed

    def _parse_condition_str(self, contigs_str):
        contigs = [c.strip() for c in contigs_str.split(",")]
        motif_residues = []

        for contig in contigs:
            if contig[0] in self.chain_alphabet:
                chain = contig[0]
                if "-" in contig:
                    motif_start, motif_end = [int(i) for i in contig[1:].split("-")]
                    for resid in range(motif_start, motif_end + 1):
                        motif_residues.append((chain, resid))
                else:
                    motif_resid = int(contig[1:])
                    motif_residues.append((chain, motif_resid))
            else:
                raise ValueError()

        # this is a little confusing
        # so motif_idx is "where we are in the motif currently"
        # and unindexed_res_idx is "what res_idx we should assign a new unindexed residue"
        # we then create "output_res_idx" which is the actual res_idx assigned to the residue
        # based on whether or not the residue is indexed

        motif_idx = 0
        unindexed_res_idx = 0
        curr_atom_idx = 0
        atoms = []
        residues = []
        atom_noising_mask = []
        redesign = []
        chain_ids = []
        chain_data = {}
        source_residue_to_output: dict[tuple[str, int], tuple[int, int, bool]] = {}

        for chain_id, resid in motif_residues:
            chain = self.structure[0][chain_id]
            residue = chain[resid]
            redesign_seq = resid in self.redesign_contigs[chain_id] if chain_id in self.redesign_contigs else False
            if self.motif_res_is_unindexed[motif_idx]:
                output_res_idx = unindexed_res_idx
                unindexed_res_idx += 1
            else:
                output_res_idx = self.motif_seq_idx[motif_idx]
            res_data, atom_data, _atom_noising_mask, new_atom_idx = biopython_to_boltz(
                residue, output_res_idx, curr_atom_idx,
                noise_bb=False,
                noise_tip=redesign_seq,
                noise_sidechain=redesign_seq
            )
            residues.append(res_data)
            atoms.extend(atom_data)
            atom_noising_mask.extend(_atom_noising_mask)
            redesign.append(redesign_seq)

            if chain_id in chain_ids:
                chain_data[chain_id]['res_num'] = chain_data[chain_id]['res_num'] + 1
                chain_data[chain_id]['atom_num'] = chain_data[chain_id]['atom_num'] + len(atom_data)
            else:
                chain_data[chain_id] = {
                    "name": chain_id,
                    "mol_type": const.chain_type_ids["PROTEIN"],
                    "entity_id": len(chain_ids),
                    "sym_id": len(chain_ids),
                    "asym_id": len(chain_ids),
                    "cyclic_period": 0,
                    "atom_idx": curr_atom_idx,
                    "atom_num": len(atom_data),
                    "res_idx": motif_idx, # output_res_idx,
                    "res_num": 1
                }
                chain_ids.append(chain_id)

            chain_asym_id = int(chain_data[chain_id]["asym_id"])
            source_residue_to_output[(chain_id, int(resid))] = (
                chain_asym_id,
                int(output_res_idx),
                bool(self.motif_res_is_unindexed[motif_idx]),
            )

            motif_idx += 1
            curr_atom_idx = new_atom_idx

        atoms = np.array(atoms, dtype=Atom)
        residues = np.array(residues, dtype=SamplingResidue)

        # # reset the chain res_idx such that the first chain starts at res_idx=0
        # min_chain_res_idx = min([c['res_idx'] for c in chain_data.values()])
        # for c in chain_data.values():
        #     c['res_idx'] = c['res_idx'] - min_chain_res_idx

        chains = np.array([astuple(ChainData(**c)) for c in chain_data.values()], dtype=Chain)

        struct = Structure(
            atoms=atoms,
            bonds=np.array([]),
            residues=residues,
            chains=chains,
            connections=np.array([]),
            interfaces=np.array([]),
            mask=np.array([True], dtype=bool)
        )
        redesign = np.array(redesign)
        atom_noising_mask = np.array(atom_noising_mask)

        return struct, atom_noising_mask, redesign, source_residue_to_output

    @staticmethod
    def _expand_segment_spec(segment_spec: str) -> list[tuple[str, int]]:
        match = re.fullmatch(r"([A-Za-z])(\d+)(?:-(\d+))?", segment_spec.strip())
        if match is None:
            raise ValueError(f"Invalid motif segment spec: {segment_spec}")
        chain_id = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) is not None else start
        if end < start:
            raise ValueError(f"Invalid segment range in motif_groups: {segment_spec}")
        return [(chain_id, resid) for resid in range(start, end + 1)]

    def _build_explicit_group_by_copy_key(
        self,
        source_residue_to_output: dict[tuple[str, int], tuple[int, int, bool]],
        sample_length: int,
    ) -> dict[tuple[int, int], int] | None:
        motif_groups = self.kwargs.get("motif_groups")
        if motif_groups is None:
            return None

        group_by_copy_key: dict[tuple[int, int], int] = {}
        for group_idx, group_segments in enumerate(motif_groups, start=1):
            if not isinstance(group_segments, Sequence) or isinstance(group_segments, (str, bytes)):
                raise ValueError("motif_groups must be a list of groups; each group is a list of segment specs")
            for segment_spec in group_segments:
                if not isinstance(segment_spec, str):
                    raise ValueError("Each motif_groups segment must be a string like A92-99 or A14")
                for source_key in self._expand_segment_spec(segment_spec):
                    if source_key not in source_residue_to_output:
                        raise ValueError(f"motif_groups segment {segment_spec} references missing residue {source_key}")
                    asym_id, output_res_idx, is_unindexed = source_residue_to_output[source_key]
                    final_res_idx = int(output_res_idx + (sample_length if is_unindexed else 0))
                    copy_key = (int(asym_id), final_res_idx)
                    if copy_key in group_by_copy_key and group_by_copy_key[copy_key] != group_idx:
                        raise ValueError(
                            f"Residue {source_key} is assigned to multiple motif groups"
                        )
                    group_by_copy_key[copy_key] = int(group_idx)
        return group_by_copy_key


    def sample_data(self):
        for _ in range(self.num_samples):

            if self.sample_contigs_idx_config is not None:
                contigs_idx = self._sample_idx_str(self.sample_contigs_idx_config)
                print(contigs_idx, self.total_len)

                max_idx = max([int(i) for i in contigs_idx.split(",") if len(i) > 0])
                max_len_str = str(max(self.total_len))
                assert max_idx < max(self.total_len), (
                    "the idx contig provided " +
                    self.sample_contigs_idx_config +
                    " is capable of sampling indicies for the motif " +
                    "which are outside the max total length for the protein " +
                    max_len_str
                )
            else:
                contigs_idx = self.contigs_idx

            if len(self.total_len) == 2:
                if len(contigs_idx) > 0:
                    max_idx = max([int(i) for i in contigs_idx.split(",")])
                else:
                    max_idx = self.num_motif_res
                min_len = max(self.total_len[0], max_idx)
                sample_length = np.random.randint(min_len, self.total_len[1] + 1)
            else:
                sample_length = self.total_len[0]

            chain_lens = {
                self.sample_chain_name: sample_length
            }

            try:
                self.motif_seq_idx, self.motif_res_is_unindexed = self._parse_idx_str(contigs_idx, self.pdb_contigs)
            except Exception as e:
                print(f"error with parsing idx contigs {self.contigs_idx}")
                raise e

            try:
                (
                    self.condition_struct,
                    self.atom_noising_mask,
                    self.res_type_noising_mask,
                    source_residue_to_output,
                ) = self._parse_condition_str(self.pdb_contigs)
            except Exception as e:
                print(f"error with parsing contigs PDB contigs {self.pdb_contigs}")
                raise e

            struct = generate_protein_structure_template(
                chain_lens,
                extra_mols=self.condition_struct,
                extra_mols_residue_is_unindexed_mask=self.motif_res_is_unindexed
            )

            # we concat on dummy masks for the new struct residues
            # since the generated residue masks only apply to motif residues
            residue_is_unindexed_mask = np.concatenate([
                np.zeros(sample_length, dtype=bool),
                self.motif_res_is_unindexed,
            ], axis=0)
            res_type_noising_mask = np.concatenate([
                np.ones(sample_length, dtype=bool),
                self.res_type_noising_mask
            ], axis=0)

            task_data = {
                "atom_noising_mask": self.atom_noising_mask,
                "res_type_noising_mask": res_type_noising_mask,
                "residue_is_unindexed_mask": residue_is_unindexed_mask,
                "motif_group_by_copy_key": self._build_explicit_group_by_copy_key(
                    source_residue_to_output=source_residue_to_output,
                    sample_length=sample_length,
                ),
                "multi_motif_se3": self.multi_motif_se3,
                "multi_motif_se3_trans_std": self.multi_motif_se3_trans_std,
                "motif_t_lock": self.motif_t_lock,
            }

            sample_metadata = build_motif_scaffolding_sample_metadata(
                source_residue_to_output=source_residue_to_output,
                motif_group_by_copy_key=task_data["motif_group_by_copy_key"],
                sample_length=sample_length,
            )

            token_data, rigid_data, token_bonds, fixed_rigids_com = sample_noise_from_struct_template(
                struct,
                task_masks=task_data
            )
            struct.atoms['coords'] -= fixed_rigids_com[None]

            data = Tokenized(
                tokens=token_data,
                rigids=rigid_data,
                bonds=token_bonds,
                structure=struct
            )

            rigids_noising_mask = rigid_data['rigids_noising_mask']
            seq_noising_mask = token_data['seq_noising_mask']

            task_data = {
                "t": np.array([1.0], dtype=float),
                "rigids_noising_mask": rigids_noising_mask,
                "seq_noising_mask": seq_noising_mask,
                "copy_indexed_token_mask": None,
                "copy_unindexed_token_mask": None,
                "multi_motif_se3": self.multi_motif_se3,
                "multi_motif_se3_trans_std": self.multi_motif_se3_trans_std,
                "motif_t_lock": self.motif_t_lock,
            }
            yield featurize_inference(
                data,
                task_data,
                task_name=self.kwargs.get("name", self.task_name),
                task_metadata=build_motif_scaffolding_task_metadata(self),
                sample_metadata=sample_metadata,
            )
