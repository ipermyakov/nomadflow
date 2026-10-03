import os
import pathlib
script_dir = pathlib.Path(__file__).parent.resolve()

from nomadflow.model.denoiser.bb.frames import DynamicGraphIpaFrameDenoiser
from nomadflow.model.denoiser.protein.frames_seq import DynamicGraphIpaFrameSeqDenoiser
from nomadflow.model.denoiser.protein.frames_seq_multichi import DynamicGraphIpaFrameSeqMultiChiDenoiser

from nomadflow.data.datasets.datamodule import FramediffDataModule

from nomadflow.stoch_interp.interpolate.se3 import SE3Interpolant, SE3InterpolantConfig
from nomadflow.stoch_interp.interpolate.protein import (
    ProteinDirichletInterpolant,
    ProteinDirichletMultiChiInterpolant,
    ProteinFisherInterpolant,
    ProteinFisherMultiChiInterpolant,
    ProteinInterpolant
)

from nomadflow.tasks.task import single_task_sampler
from nomadflow.tasks.fm.bb import BackboneFrameInterpolation
from nomadflow.tasks.fm.protein import (
    ProteinInterpolation,
    ProteinSeqInterpolation,
    ProteinSeqMultiChiInterpolation
)

TASKS = {
    "bb_fm": BackboneFrameInterpolation(SE3Interpolant(SE3InterpolantConfig())),
    "protein_fm": ProteinInterpolation(ProteinInterpolant(SE3InterpolantConfig())),
    "protein_fisher": ProteinSeqInterpolation(ProteinFisherInterpolant(SE3InterpolantConfig())),
    "protein_multichi_fisher": ProteinSeqMultiChiInterpolation(ProteinFisherMultiChiInterpolant(SE3InterpolantConfig())),
    "protein_dirichlet": ProteinSeqInterpolation(ProteinDirichletInterpolant(SE3InterpolantConfig())),
    "protein_multichi_dirichlet": ProteinSeqMultiChiInterpolation(ProteinDirichletMultiChiInterpolant(SE3InterpolantConfig())),
}

MODELS = {
    "bb_fm": DynamicGraphIpaFrameDenoiser,
    "protein_fisher": DynamicGraphIpaFrameSeqDenoiser,
    "protein_multichi_fisher": DynamicGraphIpaFrameSeqMultiChiDenoiser,
    "protein_dirichlet": DynamicGraphIpaFrameSeqDenoiser,
    "protein_multichi_dirichlet": DynamicGraphIpaFrameSeqMultiChiDenoiser,
}

def get_data(task):
    os.chdir(script_dir)
    datamodule = FramediffDataModule(
        task_sampler=single_task_sampler(
            task
        ),
        data_dir=os.path.join(script_dir, "data"),
        batch_size=10000,
        num_workers=2,
        min_ordered_percent=0.5
    )
    data = next(iter(datamodule.train_dataloader()))
    return data
