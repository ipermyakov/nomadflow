"""Python port of Flowfusion.jl and the ForwardBackward.jl pieces it calls.

Two array backends:

- NumPy arrays run on CPU only.
- PyTorch tensors run on CPU or CUDA. Pass ``device="cuda"`` to ``State.torch``
  when a GPU is available. Every sample and loss uses the device of the state,
  so the same call works on either device.

Array layout matches Julia: feature axes are first, the batch axis is last, and
the category axis of a discrete distribution is axis 0. Category labels are
0-based. In Julia they are 1-based, so a Julia mask token ``K`` is ``K - 1`` here.

Conditioning masks follow the Julia implementation: ``cmask == 0`` pins the
value, ``cmask == 1`` leaves it free.
"""

from flowfusion.batching import element, regroup, zerostate
from flowfusion.bridge import Tracker, batch, bridge, gen, resolve_prediction, stack_tracker, step
from flowfusion.cat import cat_shape, random_discrete_cat, random_literal_cat
from flowfusion.doob import doob_guide, make_guide
from flowfusion.indel import combine, lastsize, selectlastdim
from flowfusion.loss import floss, scalefloss, scaledmaskedmean
from flowfusion.manifolds import (
    Euclidean,
    SpecialOrthogonal,
    Torus,
    apply_tangent_coordinates,
    endpoint_guide,
    so3_tangent_coordinates_stack,
    tangent_guide,
)
from flowfusion.mask import mask
from flowfusion.processes import (
    BetaSchedule,
    BrownianMotion,
    Deterministic,
    DistInterpolatingDiscreteFlow,
    DistNoisyInterpolatingDiscreteFlow,
    DoobMatchingFlow,
    FProcess,
    GeneralDiscrete,
    InterpolatingDiscreteFlow,
    ManifoldProcess,
    NoisyInterpolatingDiscreteFlow,
    OUBridgeDistVarSched,
    OUBridgeExpVar,
    OUFlow,
    OrnsteinUhlenbeck,
    OrnsteinUhlenbeckExpVar,
    PiQ,
    ScheduledGaussianMixture,
    UniformDiscrete,
    UniformUnmasking,
)
from flowfusion.propagation import (
    backward,
    endpoint_conditioned_sample,
    forward,
    interpolate,
    likelihood_product,
)
from flowfusion.states import (
    CategoricalLikelihood,
    ContinuousState,
    DiscreteState,
    GaussianLikelihood,
    Guide,
    ManifoldState,
    MaskedState,
    dense,
    onehot,
    stochastic,
    tensor,
    unhot,
    unmask,
)

__all__ = [
    "BetaSchedule",
    "BrownianMotion",
    "CategoricalLikelihood",
    "ContinuousState",
    "Deterministic",
    "DiscreteState",
    "DistInterpolatingDiscreteFlow",
    "DistNoisyInterpolatingDiscreteFlow",
    "DoobMatchingFlow",
    "Euclidean",
    "FProcess",
    "GaussianLikelihood",
    "GeneralDiscrete",
    "Guide",
    "InterpolatingDiscreteFlow",
    "ManifoldProcess",
    "ManifoldState",
    "MaskedState",
    "NoisyInterpolatingDiscreteFlow",
    "OUBridgeDistVarSched",
    "OUBridgeExpVar",
    "OUFlow",
    "OrnsteinUhlenbeck",
    "OrnsteinUhlenbeckExpVar",
    "PiQ",
    "ScheduledGaussianMixture",
    "SpecialOrthogonal",
    "Torus",
    "Tracker",
    "UniformDiscrete",
    "UniformUnmasking",
    "apply_tangent_coordinates",
    "backward",
    "batch",
    "bridge",
    "cat_shape",
    "combine",
    "dense",
    "doob_guide",
    "element",
    "endpoint_conditioned_sample",
    "endpoint_guide",
    "floss",
    "forward",
    "gen",
    "interpolate",
    "lastsize",
    "likelihood_product",
    "make_guide",
    "mask",
    "onehot",
    "random_discrete_cat",
    "random_literal_cat",
    "regroup",
    "resolve_prediction",
    "scalefloss",
    "scaledmaskedmean",
    "selectlastdim",
    "so3_tangent_coordinates_stack",
    "stack_tracker",
    "step",
    "stochastic",
    "tangent_guide",
    "tensor",
    "unhot",
    "unmask",
    "zerostate",
]

__version__ = "0.1.5"
