
from hydra_zen import load_from_yaml
from torch.utils.data import Dataset # IterableDataset
from torch_geometric.data import Batch

from .task import SamplingTask
from .unconditional import UnconditionalSampling
from .unconditional_smiles import UnconditionalSamplingFromSMILES
from .motif_scaffolding import MotifScaffoldingTask


class BiomoleculeTaskDispatcher(Dataset):
    name_to_task_class = {
        "motif_scaffolding": MotifScaffoldingTask,
        "unconditional": UnconditionalSampling,
        "unconditional_smiles": UnconditionalSamplingFromSMILES
    }

    def __init__(
        self,
        tasks_yaml,
        batch_size,
        batching_mode="optimal",
        use_collate_for_pad=False
    ):
        super().__init__()
        assert batching_mode in ["lazy", "optimal"]
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.use_collate_for_pad = use_collate_for_pad
        self.task_objs = []
        self.task_configs = []

        self.config = load_from_yaml(tasks_yaml)

        for task_dict in self.config['tasks']:
            task_class = self.name_to_task_class[task_dict['task']]
            task = task_class(**task_dict)
            self.task_objs.append((task_dict, task))

        self.batches = self._optimal_batching()

    def __iter__(self):
        return iter(self.batches)

    def __getitem__(self, index):
        return self.batches[index]

    def __len__(self):
        return len(self.batches)

    def _optimal_batching(self):
        all_samples = []
        for _, task in self.task_objs:
            for sample in task.sample_data():
                all_samples.append(sample)
        all_samples = sorted(all_samples, key=lambda data: data['token']['token_idx'].numel())

        # batches = []
        # current_batch = []
        # max_sample_size = 0
        # for sample in all_samples:
        #     sample_size = (sample['residue']['num_nodes']) ** 2
        #     _max_sample_size = max(sample_size, max_sample_size)

        #     if _max_sample_size * len(current_batch) <= self.batch_size:
        #         current_batch.append(sample)
        #         max_sample_size = _max_sample_size
        #     else:
        #         batches.append(current_batch)
        #         current_batch = [sample]
        #         max_sample_size = sample_size
        # if len(current_batch) > 0:
        #     batches.append(current_batch)

        return all_samples