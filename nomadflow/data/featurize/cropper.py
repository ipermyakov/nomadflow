"""
Adapted from boltz
https://github.com/jwohlwend/boltz/blob/main/src/boltz/data/crop/cropper.py
"""
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from nomadflow.boltz.data import const
from nomadflow.data.featurize.tokenize import Tokenized


def _reindex_cropped_tokenized(
    data: Tokenized,
    token_data: np.ndarray,
    token_bonds: np.ndarray,
) -> Tokenized:
    tokens = token_data.copy()
    old_to_new_token_idx = {
        int(token["token_idx"]): new_token_idx for new_token_idx, token in enumerate(tokens)
    }

    rigid_blocks = []
    next_rigid_idx = 0
    for new_token_idx, token in enumerate(tokens):
        old_rigid_start = int(token["rigid_idx"])
        rigid_num = int(token["rigid_num"])
        old_rigid_end = old_rigid_start + rigid_num

        token["token_idx"] = new_token_idx
        token["rigid_idx"] = next_rigid_idx

        rigid_block = data.rigids[old_rigid_start:old_rigid_end].copy()
        rigid_block["token_idx"] = new_token_idx
        rigid_block["rigid_idx"] = np.arange(next_rigid_idx, next_rigid_idx + rigid_num)
        rigid_blocks.append(rigid_block)
        next_rigid_idx += rigid_num

    if rigid_blocks:
        rigids = np.concatenate(rigid_blocks, axis=0)
    else:
        rigids = data.rigids[:0].copy()

    bonds = token_bonds.copy()
    if len(bonds):
        keep_mask = np.isin(bonds["token_1"], list(old_to_new_token_idx)) & np.isin(
            bonds["token_2"], list(old_to_new_token_idx)
        )
        bonds = bonds[keep_mask]
        if len(bonds):
            bonds["token_1"] = np.array(
                [old_to_new_token_idx[int(token_idx)] for token_idx in bonds["token_1"]],
                dtype=bonds["token_1"].dtype,
            )
            bonds["token_2"] = np.array(
                [old_to_new_token_idx[int(token_idx)] for token_idx in bonds["token_2"]],
                dtype=bonds["token_2"].dtype,
            )

    return replace(data, tokens=tokens, rigids=rigids, bonds=bonds)


def pick_random_token(
    tokens: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    return tokens[random.randint(len(tokens))]


def pick_chain_token(
    tokens: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from a chain.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain_tokens = tokens[tokens["asym_id"] == chain_id]

    # Pick from chain, fallback to all tokens
    if chain_tokens.size:
        query = pick_random_token(chain_tokens, random)
    else:
        query = pick_random_token(tokens, random)

    return query


def pick_interface_token(
    tokens: np.ndarray,
    interface: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from an interface.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    interface : int
        The interface ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Sample random interface
    chain_1 = int(interface["chain_1"])
    chain_2 = int(interface["chain_2"])

    tokens_1 = tokens[tokens["asym_id"] == chain_1]
    tokens_2 = tokens[tokens["asym_id"] == chain_2]

    # If no interface, pick from the chains
    if tokens_1.size and (not tokens_2.size):
        query = pick_random_token(tokens_1, random)
    elif tokens_2.size and (not tokens_1.size):
        query = pick_random_token(tokens_2, random)
    elif (not tokens_1.size) and (not tokens_2.size):
        query = pick_random_token(tokens, random)
    else:
        # If we have tokens, compute distances
        tokens_1_coords = tokens_1["center_coords"]
        tokens_2_coords = tokens_2["center_coords"]

        dists = cdist(tokens_1_coords, tokens_2_coords)
        cuttoff = dists < const.interface_cutoff

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (const.interface_cutoff + 5.0)

        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Select random token
        candidates = np.concatenate([tokens_1, tokens_2])
        query = pick_random_token(candidates, random)

    return query


def pick_crop_query_token(
    data: Tokenized,
    valid_tokens: np.ndarray,
    random: np.random.RandomState,
    *,
    chain_id: Optional[int] = None,
    interface_id: Optional[int] = None,
) -> np.ndarray:
    """Apply the shared chain/interface/random crop-query semantics."""
    chains = data.structure.chains
    mask = data.structure.mask
    interfaces = data.structure.interfaces
    valid_chains = chains[mask]
    valid_interfaces = interfaces
    valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
    valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]
    try:
        if chain_id is not None:
            return pick_chain_token(valid_tokens, chain_id, random)
        if interface_id is not None:
            return pick_interface_token(valid_tokens, interfaces[interface_id], random)
        if (~data.rigids["rigids_noising_mask"]).any():
            fixed_mask = (
                ~data.rigids["rigids_noising_mask"] & data.rigids["is_present"]
            )
            fixed_rigids = data.rigids[fixed_mask]
            com_coord = fixed_rigids["tensor7"][..., 4:].mean(axis=0)
            squared_distance = np.sum(
                (fixed_rigids["tensor7"][..., 4:] - com_coord[None]) ** 2,
                axis=-1,
            )
            weights = 1 / (squared_distance ** 1.5 + 1e-5)
            rigid = random.choice(fixed_rigids, p=weights / weights.sum())
            return data.tokens[int(rigid["token_idx"])]
        if valid_interfaces.size:
            interface = valid_interfaces[random.randint(len(valid_interfaces))]
            return pick_interface_token(valid_tokens, interface, random)
        chosen_chain = valid_chains[random.randint(len(valid_chains))]["asym_id"]
        return pick_chain_token(valid_tokens, chosen_chain, random)
    except Exception as error:
        print(f"ran into cropping error {error}, just doing random cropping")
        chosen_chain = valid_chains[random.randint(len(valid_chains))]["asym_id"]
        return pick_chain_token(valid_tokens, chosen_chain, random)


def filter_tokenized_to_resolved(data: Tokenized) -> Tokenized:
    """Drop unresolved scaffold tokens without performing an additional crop."""
    token_data = data.tokens[data.tokens["resolved_mask"]]
    return _reindex_cropped_tokenized(data, token_data, data.bonds)


class Cropper:
    """Interpolate between contiguous and spatial crops."""

    def __init__(self, min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_rigids: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
        required_token_indices: Optional[np.ndarray] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.
        required_token_indices : np.ndarray, optional
            Token indices that must be retained. These count towards
            ``max_tokens`` and ``max_rigids``.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Check inputs
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)

        # Randomly select a neighborhood size
        neighborhood_size = random.choice(self.neighborhood_sizes)

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        required_indices = {
            int(token_idx)
            for token_idx in (
                np.asarray(required_token_indices).reshape(-1)
                if required_token_indices is not None
                else []
            )
        }
        available_indices = {int(token_idx) for token_idx in token_data["token_idx"]}
        missing_required = required_indices - available_indices
        if missing_required:
            msg = f"Required token indices are not present: {sorted(missing_required)}"
            raise ValueError(msg)
        if len(required_indices) > max_tokens:
            msg = (
                f"{len(required_indices)} required tokens exceed "
                f"the crop budget of {max_tokens}"
            )
            raise ValueError(msg)

        required_tokens = token_data[
            np.isin(token_data["token_idx"], list(required_indices))
        ]
        required_rigids = int(np.sum(required_tokens["rigid_num"]))
        if max_rigids is not None and required_rigids > max_rigids:
            msg = (
                f"{required_rigids} rigids from required tokens exceed "
                f"the crop budget of {max_rigids}"
            )
            raise ValueError(msg)

        query = pick_crop_query_token(
            data,
            valid_tokens,
            random,
            chain_id=chain_id,
            interface_id=interface_id,
        )

        # Sort all tokens by distance to query_coords
        dists = valid_tokens["center_coords"] - query["center_coords"]
        indices = np.argsort(np.linalg.norm(dists, axis=1))

        # Select cropped indices
        cropped: set[int] = set(required_indices)
        total_rigids = required_rigids
        for idx in indices:
            # Get the token
            token = valid_tokens[idx]

            # Get all tokens from this chain
            chain_tokens = valid_tokens[
                valid_tokens["asym_id"] == token["asym_id"]
            ]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= neighborhood_size:
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighborhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                min_idx = token["res_idx"] - neighborhood_size
                max_idx = token["res_idx"] + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                # Start by adding just the query token
                new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                #
                # max_token_set is already window-limited; if it has fewer than
                # neighborhood_size resolved tokens (gaps / sparse res_idx),
                # expanding min/max cannot grow the set — stop to avoid hang.
                min_idx = max_idx = int(token["res_idx"])
                while new_tokens.size < neighborhood_size:
                    if new_tokens.size >= len(max_token_set):
                        break
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    expanded = max_token_set[
                        (max_token_set["res_idx"] >= min_idx)
                        & (max_token_set["res_idx"] <= max_idx)
                    ]
                    if expanded.size == new_tokens.size:
                        break
                    new_tokens = expanded

            # Compute new tokens and new atoms
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_tokens = token_data[list(new_indices)]
            new_rigids = np.sum(new_tokens["rigid_num"])

            # Stop if we exceed the max number of tokens or atoms
            if (len(new_indices) > (max_tokens - len(cropped))) or (
                (max_rigids is not None) and ((total_rigids + new_rigids) > max_rigids)
            ):
                break

            # Add new indices
            cropped.update(new_indices)
            total_rigids += new_rigids

        # Get the cropped tokens sorted by index
        token_data = token_data[np.isin(token_data["token_idx"], sorted(cropped))]
        return _reindex_cropped_tokenized(data, token_data, token_bonds)