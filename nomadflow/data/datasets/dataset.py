""" Generic dataset for all input data """
from typing import Dict
import logging
import os
import json

import torch
import torch.utils.data as data

from nomadflow.boltz.data import const
from nomadflow.boltz.data.types import Record


class MMCIFDataset(data.Dataset):
    def __init__(
            self,
            data_dir,
            data_sampler,
            task_sampler,
            min_num_res=30,
            max_num_res=5000,
            min_percent_ordered=0.5,
            max_resolution=3.0,
            subset=None,
            split=None,
            exclude_mol_types=None,
            overfit_num=None,
            count_on_protein_res=False
        ):
        self._log = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.min_num_res = min_num_res
        self.max_num_res = max_num_res
        self.subset = subset
        self.split = split
        self.min_percent_ordered = min_percent_ordered
        self.max_resolution = max_resolution
        self.task_sampler = task_sampler
        self.data_sampler = data_sampler
        self.overfit_num = overfit_num
        self.count_on_protein_res = count_on_protein_res
        if exclude_mol_types is None:
            self.exclude_mol_types = []
        else:
            print("Excluding molecules that contain chains of types:", exclude_mol_types)
            self.exclude_mol_types = [const.chain_types.index(s) for s in exclude_mol_types]

        self._init_metadata()

        print(f"Training on {len(self.manifest)} datapoints")

    def _init_metadata(self):
        """Initialize metadata."""
        manifest_path = os.path.join(self.data_dir, "manifest.json")

        # Process CSV with different filtering criterions.
        with open(manifest_path) as fp:
            self.raw_manifest = json.load(fp)

        self.manifest = []
        for record in self.raw_manifest:
            # remove records which contain mol types we're excluding
            _exclude_record = False
            for chain in record['chains']:
                if chain['mol_type'] in self.exclude_mol_types:
                    _exclude_record = True
                    break
            if _exclude_record:
                continue
            # apply some filtering critera that we might change at train time
            if self.count_on_protein_res:
                num_res = sum(chain['num_residues'] for chain in record['chains'] if chain['mol_type'] == 0)
            else:
                num_res = sum(chain['num_residues'] for chain in record['chains'])
            if num_res > self.max_num_res or num_res < self.min_num_res:
                continue
            resolution = record['structure']['resolution']
            if resolution is not None and resolution > self.max_resolution:
                continue
            if self.split is not None and record['id'] not in self.split:
                continue
            # if record['structure']['coil_percent'] > (1 - self.min_percent_ordered):
            #     continue

            record['is_af2_struct'] = record['id'].startswith("AF-")

            self.manifest.append(Record.from_dict(record))

        if self.overfit_num is not None:
            self.manifest = self.manifest[-self.overfit_num:]
            print("overfit entries:", self.manifest)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # Sample data example.
        record = self.manifest[idx]
        return record


class LengthDataset(data.Dataset):
    def __init__(
            self,
            lengths: Dict[int, int],
            batch_size: int = 3000,
            same_length_per_batch=False,
            normalize_cache_path=None,
            batch_by_edge_fn=None,
        ):
        self._log = logging.getLogger(__name__)
        self.lengths = lengths
        self.batch_size = batch_size
        if normalize_cache_path is not None:
            self.cache_stats = torch.load(normalize_cache_path)
        else:
            self.cache_stats = None

        self.batches = []
        self.sample_ids = []
        current_batch = []
        current_sample_ids = []
        sample_id = 0
        for sample_l, num_samples in self.lengths.items():
            for _ in range(num_samples):
                if batch_by_edge_fn is not None:
                    current_batch_size = sum([batch_by_edge_fn(l) for l in current_batch])
                else:
                    current_batch_size = sum(current_batch)
                if current_batch_size >= self.batch_size:
                    self.batches.append(current_batch)
                    self.sample_ids.append(current_sample_ids)
                    current_batch = []
                    current_sample_ids = []
                current_batch.append(sample_l)
                current_sample_ids.append(sample_id)
                sample_id += 1
            if same_length_per_batch:
                self.batches.append(current_batch)
                self.sample_ids.append(current_sample_ids)
                current_batch = []
                current_sample_ids = []

        if len(current_batch) > 0:
            self.batches.append(current_batch)
        if len(current_sample_ids) > 0:
            self.sample_ids.append(current_sample_ids)


    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        ret =  {
            "num_res": self.batches[idx],
            "sample_id": self.sample_ids[idx]
        }
        if self.cache_stats is not None:
            ret['latent_norm_mu'] = self.cache_stats['mu'].clone()
            ret['latent_norm_std'] = self.cache_stats['std'].clone()

        return ret
