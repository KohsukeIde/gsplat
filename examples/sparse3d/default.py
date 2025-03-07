from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops import remove, reset_opa  # <- duplicate, split を使わなくなるので import外してもOK
from typing_extensions import Literal


@dataclass
class DefaultStrategy(Strategy):
    """
    A modified DefaultStrategy that does NOT perform any duplication, splitting,
    pruning, or opacity reset. This means the number of Gaussians is fixed
    at the initial count (init_num_pts) throughout training.

    All the methods related to duplication/splitting/pruning are kept,
    but they are never invoked in `step_post_backward()`.

    Args:
        prune_opa (float): Unused in this modified version.
        grow_grad2d (float): Unused in this modified version.
        grow_scale3d (float): Unused in this modified version.
        grow_scale2d (float): Unused in this modified version.
        prune_scale2d (float): Unused in this modified version.
        refine_scale2d_stop_iter (int): Unused in this modified version.
        refine_start_iter (int): Unused in this modified version.
        refine_stop_iter (int): Unused in this modified version.
        reset_every (int): Unused in this modified version.
        refine_every (int): Unused in this modified version.
        pause_refine_after_reset (int): Unused in this modified version.
        absgrad (bool): Unused in this modified version.
        revised_opacity (bool): Unused in this modified version.
        verbose (bool): Unused in this modified version.
        key_for_gradient (str): Which variable uses for densification strategy,
            unused in this version since we do not densify.

    Note:
        - We still keep `_update_state()` to gather gradient stats, but it's only
          for reference. We do not use them to modify the Gaussians.

    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05

    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """
        Initialize and return the running state for this strategy.
        Here we do not do duplication/pruning, so we only store placeholders.
        """
        return {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
            "radii": None,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """
        Sanity check for the parameters and optimizers.
        We still require 'means', 'scales', 'quats', 'opacities' exist, 
        but we do not check or use duplication/splitting/pruning keys.
        """
        super().check_sanity(params, optimizers)
        # The following keys are required for minimal 3DGS param set.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """
        Callback before loss.backward().
        We do nothing special here, except we keep code to ensure
        'info' has the correct gradient field if needed.
        """
        if self.key_for_gradient in info:
            info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """
        Callback after loss.backward().
        In the original code, duplication / splitting / pruning / reset
        happen here. We remove all those calls to keep #Gaussians fixed.

        We only update some internal stats via `_update_state()`,
        but do not act on them.
        """
        # Optionally gather some gradient stats,
        # though we do not use them for densification:
        self._update_state(params, state, info, packed=packed)

        # Do NOT call duplication/splitting/pruning/reset. 
        # So #Gaussians never changes.

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """
        We keep this just in case we want to log or store gradient stats.
        But we never use them to refine or prune.
        """
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            if key not in info:
                # It's OK if they're missing, we just won't do anything
                return

        # We do minimal logic here:
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if state["radii"] is None:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        if packed:
            gs_ids = info["gaussian_ids"]
            radii = info["radii"]
            # grads: [nnz, 2]
        else:
            sel = info["radii"] > 0.0
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]           # [nnz, 2]
            radii = info["radii"][sel]   # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        # Just store the maximum radius for reference
        state["radii"][gs_ids] = torch.maximum(
            state["radii"][gs_ids],
            radii / float(max(info["width"], info["height"])),
        )
