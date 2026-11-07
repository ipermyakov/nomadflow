# flowfusion

Python port of [Flowfusion.jl](https://github.com/MurrellGroup/Flowfusion.jl), including the
[ForwardBackward.jl](https://github.com/MurrellGroup/ForwardBackward.jl) processes it samples from.

NumPy arrays are the CPU backend. PyTorch tensors are the CPU and CUDA backend: sampling,
bridges, and losses use the device of the state you pass in. Move a state with
`state.torch("cuda")` or `state.torch("cpu")`. `state.numpy()` returns the CPU array.

```python
import numpy as np
from flowfusion import BrownianMotion, ContinuousState, Deterministic, bridge, floss, gen, scalefloss

P = Deterministic()                 # or BrownianMotion(0.15)
X0 = ContinuousState(np.random.randn(2, 128).astype(np.float32))
X1 = ContinuousState(np.random.randn(2, 128).astype(np.float32) + 2)
t = np.random.rand(128).astype(np.float32)
Xt = bridge(P, X0, X1, t)
# train a model so that floss(P, model(t, Xt), X1, scalefloss(P, t)) is small
# samples = gen(P, X0, model, np.linspace(0, 1, 101))
```

States use Julia's layout: features first, batch last. Discrete labels are 0-based
(Julia is 1-based). A one-hot tensor has its category axis at position 0.
`cmask == 0` pins a value to the conditional observation; `cmask == 1` leaves it free.

Euclidean space, the flat torus, and SO(3) are implemented. Other `Manifolds.jl`
types are not.

See `examples/flow_matching_demo.ipynb` for a small 2D flow-matching model.
