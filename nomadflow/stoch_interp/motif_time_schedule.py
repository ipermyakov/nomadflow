"""Motif corruption / integration time schedules for early lock (t_lock).

Global corruption time ``t`` runs in [0, 1] (noise → data). When ``motif_t_lock``
is set, motif group motion uses a schedule to map ``t`` into motif interpolation
time in [0, 1], and may clamp motif rigids to ground truth once locked.

Swap ``motif_time_schedule`` on :class:`~nomadflow.stoch_interp.multiframe.MultiSE3Interpolant`
to plug in non-linear curves without changing corruption call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union

import torch

MotifTLock = Union[float, torch.Tensor]


def _broadcast_t_lock(t_lock: MotifTLock, t: torch.Tensor) -> torch.Tensor:
    """Scalar or per-batch ``t_lock`` broadcast to ``t``'s shape."""
    t_lock_t = torch.as_tensor(t_lock, device=t.device, dtype=t.dtype)
    if t_lock_t.ndim == 0:
        return t_lock_t
    while t_lock_t.ndim < t.ndim:
        t_lock_t = t_lock_t.unsqueeze(-1)
    return t_lock_t


class MotifTimeSchedule(Protocol):
    """Maps global batch time to motif group-motion time and lock flags."""

    def motif_interp_time(
        self,
        t: torch.Tensor,
        t_lock: MotifTLock,
    ) -> torch.Tensor:
        """Motif interpolation coordinate in [0, 1] for group SE(3) motion."""

    def is_locked(
        self,
        t: torch.Tensor,
        t_lock: MotifTLock,
    ) -> torch.Tensor:
        """Per-batch-element mask: motif rigids should equal ``x_1``."""


@dataclass(frozen=True)
class LinearMotifTimeSchedule:
    """Linear group motion on [0, t_lock]; clamp to GT for ``t >= t_lock``.

    Group motion uses ``t_motif = min(t, t_lock) / t_lock``, so the motif reaches
    GT exactly at global ``t == t_lock``. For ``t > t_lock``, ``is_locked`` is True
    and corruption forces ``x_t = x_1``.
    """

    def motif_interp_time(
        self,
        t: torch.Tensor,
        t_lock: MotifTLock,
    ) -> torch.Tensor:
        t_lock_t = _broadcast_t_lock(t_lock, t)
        disabled = t_lock_t <= 0
        interp = (torch.minimum(t, t_lock_t) / t_lock_t.clamp_min(1e-12)).clamp(0.0, 1.0)
        return torch.where(disabled, torch.ones_like(interp), interp)

    def is_locked(
        self,
        t: torch.Tensor,
        t_lock: MotifTLock,
    ) -> torch.Tensor:
        t_lock_t = _broadcast_t_lock(t_lock, t)
        return t >= t_lock_t


_MOTIF_TIME_SCHEDULE_REGISTRY: dict[str, type[LinearMotifTimeSchedule]] = {
    "linear": LinearMotifTimeSchedule,
}


def build_motif_time_schedule(name: Optional[str]) -> Optional[MotifTimeSchedule]:
    if name is None:
        return None
    key = name.lower()
    if key in ("none", "off", "disabled"):
        return None
    try:
        cls = _MOTIF_TIME_SCHEDULE_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_MOTIF_TIME_SCHEDULE_REGISTRY))
        raise ValueError(
            f"Unknown motif_time_schedule {name!r}; known: {known}"
        ) from exc
    return cls()


# Sentinel in featurized batches when early lock is off (see assembler.MOTIF_T_LOCK_DISABLED).
MOTIF_T_LOCK_DISABLED = -1.0


def motif_t_lock_is_enabled(motif_t_lock: MotifTLock | None) -> bool:
    """True when ``motif_t_lock`` activates :meth:`MotifTimeSchedule.is_locked`."""
    if motif_t_lock is None:
        return False
    t_lock_t = torch.as_tensor(motif_t_lock)
    if t_lock_t.numel() == 0:
        return False
    return bool((t_lock_t > MOTIF_T_LOCK_DISABLED).any())


def effective_rigids_noising_mask(
    base_noising_mask: torch.Tensor,
    motif_group_id: torch.Tensor | None,
    t: torch.Tensor,
    motif_t_lock: MotifTLock | None,
    schedule: Optional[MotifTimeSchedule] = None,
) -> torch.Tensor:
    """Per-rigid noising mask with motif early lock (static-motif behavior when locked).

    When ``t >= motif_t_lock``, all rigids with ``motif_group_id > 0`` (real motif and
    virtual aggregators) are treated as non-noising, matching tokenized motif copies with
    ``rigids_noising_mask=False``.
    """
    base = base_noising_mask.to(dtype=torch.bool)
    if not motif_t_lock_is_enabled(motif_t_lock) or motif_group_id is None:
        return base
    if schedule is None:
        schedule = LinearMotifTimeSchedule()
    t = t.to(device=base.device)
    group_id = motif_group_id.to(device=base.device, dtype=torch.long)
    motif_rigid_mask = group_id > 0
    t_lock_t = _broadcast_t_lock(motif_t_lock, t)
    lock_enabled = t_lock_t > MOTIF_T_LOCK_DISABLED
    locked = schedule.is_locked(t, motif_t_lock) & lock_enabled
    while locked.ndim < base.ndim:
        locked = locked.unsqueeze(-1)
    locked = locked.expand_as(base)
    return base & ~(motif_rigid_mask & locked)


def stash_base_rigids_noising_mask(rigids_data) -> torch.Tensor:
    """Preserve tokenized ``rigids_noising_mask`` before motif_t_lock overrides."""
    if '_rigids_noising_mask_base' not in rigids_data:
        rigids_data['_rigids_noising_mask_base'] = rigids_data['rigids_noising_mask'].clone()
    return rigids_data['_rigids_noising_mask_base']


def apply_motif_t_lock_to_rigids_noising_mask(
    batch,
    t=None,
    motif_t_lock: MotifTLock | None = None,
    schedule: Optional[MotifTimeSchedule] = None,
) -> None:
    """Write time-dependent ``rigids_noising_mask`` on ``batch['rigids']`` (in-place).

    Call at model input (denoiser forward) and after corruption so IPA / loss see the
    same mask. Uses stashed base mask when present.
    """
    rigids_data = batch['rigids']
    t = batch['t'] if t is None else t
    if motif_t_lock is None:
        motif_t_lock = batch.get('motif_t_lock')
    rigids_data['rigids_noising_mask'] = effective_rigids_noising_mask(
        stash_base_rigids_noising_mask(rigids_data),
        rigids_data.get('motif_group_id'),
        t,
        motif_t_lock,
        schedule=schedule,
    )
