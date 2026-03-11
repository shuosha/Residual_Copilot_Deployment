"""Task-specific geometric offsets for residual shared autonomy.

Each task has offsets for the 'fixed' and 'held' objects that adjust
FoundationPose-estimated positions to the functional contact points
used by the residual policy. Offsets are in local object frames and
applied via `combine_frame_transforms`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable

import torch

from robot_control.utils.kinematics_utils import _quat_apply


def _is_z_axis_up(q: torch.Tensor) -> bool:
    """Check if z-axis of quaternion (w,x,y,z) points up in world frame."""
    z_local = torch.tensor([[0., 0., 1.]], device=q.device, dtype=q.dtype)
    z_world = _quat_apply(q, z_local)
    return bool(z_world[0, 2] > 0)


@dataclass
class ObjectOffset:
    """Offset specification for a single object.

    If ``alt_offset`` is provided, ``condition`` selects between
    ``offset`` (condition True) and ``alt_offset`` (condition False).
    """
    offset: List[float]
    alt_offset: Optional[List[float]] = None
    condition: Optional[Callable[[torch.Tensor], bool]] = None


@dataclass
class TaskOffsets:
    """Pair of object offsets for a task."""
    fixed: ObjectOffset
    held: ObjectOffset


TASK_OFFSETS: dict[str, TaskOffsets] = {
    "gearmesh": TaskOffsets(
        fixed=ObjectOffset(
            offset=[0.02025, 0.0, 0.0250],
            alt_offset=[-0.02025, 0.0, -0.02],
            condition=_is_z_axis_up,
        ),
        held=ObjectOffset(offset=[0.0, 0.0, 0.0125 - 0.005]),
    ),
    "peginsert": TaskOffsets(
        fixed=ObjectOffset(offset=[0.0, 0.0, 0.025]),
        held=ObjectOffset(
            offset=[0.0, 0.0, 0.05 - 0.02],
            alt_offset=[0.0, 0.0, 0.02],
            condition=_is_z_axis_up,
        ),
    ),
    "nutthread": TaskOffsets(
        fixed=ObjectOffset(offset=[0.0, 0.0, 0.035]),
        held=ObjectOffset(offset=[0.0, 0.0, 0.0]),
    ),
}


def apply_task_offsets(
    task_name: str,
    fixed_pos: torch.Tensor,
    fixed_quat: torch.Tensor,
    held_pos: torch.Tensor,
    held_quat: torch.Tensor,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply task-specific offsets to fixed and held object positions.

    Args:
        task_name: One of 'gearmesh', 'peginsert', 'nutthread'.
        fixed_pos: (1, 3) position of the fixed object.
        fixed_quat: (1, 4) quaternion of the fixed object (w,x,y,z).
        held_pos: (1, 3) position of the held object.
        held_quat: (1, 4) quaternion of the held object (w,x,y,z).
        device: Torch device.

    Returns:
        (fixed_pos, held_pos) with offsets applied.
    """
    from robot_control.utils.math import combine_frame_transforms

    offsets = TASK_OFFSETS[task_name]

    def _apply(pos, quat, obj_offset: ObjectOffset):
        if obj_offset.offset == [0.0, 0.0, 0.0]:
            return pos
        if obj_offset.condition is not None and obj_offset.alt_offset is not None:
            off = obj_offset.offset if obj_offset.condition(quat) else obj_offset.alt_offset
        else:
            off = obj_offset.offset
        return combine_frame_transforms(
            pos, quat,
            torch.tensor([off], dtype=torch.float32).to(device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).to(device),
        )[0]

    fixed_pos = _apply(fixed_pos, fixed_quat, offsets.fixed)
    held_pos = _apply(held_pos, held_quat, offsets.held)

    return fixed_pos, held_pos
