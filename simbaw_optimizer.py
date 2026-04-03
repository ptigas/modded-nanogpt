"""
SimbaW Optimizer - Structure based on NorMuon, algorithm from Simba paper
Combines NorMuon's efficient distributed implementation with SimbaW's preconditioning.
"""

import math
import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.optimizer import Optimizer
from collections import defaultdict


class NorSimbaW(Optimizer):
    """
    SimbaW optimizer with NorMuon-style structure.

    SimbaW uses:
    - Momentum-based gradient averaging (like NorMuon)
    - Random subsampling of dimensions (coarse_dim_perc) - KEY DIFFERENCE
    - Low-rank SVD approximation for preconditioning - KEY DIFFERENCE
    - Low-rank variance estimator (like NorMuon's second_momentum_buffer)
    - Decoupled weight decay (like both)

    Args:
        params: Parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient (decoupled)
        momentum: Momentum factor for gradient EMA
        beta2: Beta2 for second moment estimation
        coarse_dim_perc: Percentage of dimensions to sample (0.0-1.0)
        rank: Rank for truncated SVD
        eps: Epsilon for numerical stability
        custom_sizing: Use custom parameter grouping for 8 GPUs
    """

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, beta2=0.95,
                 coarse_dim_perc=0.5, rank=20, eps=1e-8, custom_sizing=True):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            beta2=beta2,
            coarse_dim_perc=coarse_dim_perc,
            rank=rank,
            eps=eps
        )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Use same parameter grouping strategy as NorMuon
        if custom_sizing and dist.get_world_size() == 8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)

        super().__init__(param_groups, defaults)

    def reset(self):
        """Clear momentum buffers"""
        for group in self.param_groups:
            if "momentum_buffer" in group:
                group["momentum_buffer"].zero_()
            if "second_momentum_buffer" in group:
                group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Standard parameter grouping for non-8-GPU setups.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Custom parameter grouping optimized for 8 GPUs.
        Matches NorMuon's grouping strategy.
        """
        module_group_order = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        params_list = list(params)
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    @torch.no_grad()
    def step(self):
        """
        Efficient distributed step following NorMuon's three-pass structure.
        Key algorithmic difference: Uses SimbaW preconditioning instead of polar_express.
        """
        rank = dist.get_rank()
        group_infos = []

        # First pass: Stack gradients and launch reduce_scatter
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params):].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))

        all_gather_infos = []

        # Second pass: Wait for gradients, compute updates, launch all_gather
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(chunk_size, max(0, len(params) - start_idx))

            # Initialize momentum buffer (first moment)
            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]

            # Apply momentum update
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape

            # Handle attn reshaping (same as NorMuon)
            if params[module_idx].label == 'attn':
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)

            ref_param = params[module_idx]
            param_shape = ref_param.shape

            # Initialize second momentum buffer (variance estimator)
            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (
                    torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1]
                    else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            # Initialize learning rate multipliers
            if "param_lr" not in group:
                group["param_lr"] = (
                    max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                    * ref_param.new_tensor(
                        [getattr(param, "lr_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                    ).view(-1, 1, 1)
                )

                group["param_wd"] = ref_param.new_tensor(
                    [getattr(param, "wd_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                ).view(-1, 1, 1)

            # Effective learning rate and weight decay
            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["weight_decay"] * group["param_wd"]

            # Apply SimbaW preconditioning (KEY ALGORITHMIC DIFFERENCE)
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = simba_precondition_compiled(
                    updated_grads,
                    coarse_dim_perc=group["coarse_dim_perc"],
                    rank=group["rank"],
                    eps=group["eps"]
                )

            # Apply variance-based step size scaling (like NorMuon)
            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(
                dim=-1 if param_shape[-2] >= param_shape[-1] else -2,
                keepdim=True
            )
            second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

            # Reshape back if needed
            v_chunk = v_chunk.view(grad_shape)

            # Prepare updated parameters
            updated_params = torch.empty_like(grad_chunk)
            param_chunk = (
                torch.stack(params[module_idx:module_idx + num_params])
                if num_params > 0
                else torch.zeros_like(v_chunk)
            )

            # Apply weight decay and update
            param_chunk.mul_(1 - eff_wd)
            param_chunk.add_(-eff_lr * v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            # Prepare for all_gather
            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            all_gather_infos.append({
                "gather_future": gather_future,
                "stacked_params": stacked_params,
                "orig_params": params,
            })

        # Third pass: Wait for all_gather and copy results back
        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)


@torch.compile(dynamic=False, fullgraph=True)
def simba_precondition_compiled(
    G: Tensor,
    coarse_dim_perc: float = 0.5,
    rank: int = 20,
    eps: float = 1e-8
) -> Tensor:
    """
    SimbaW preconditioning with dimension coarsing and low-rank SVD.
    This is the KEY ALGORITHMIC DIFFERENCE from NorMuon's polar_express.

    Args:
        G: Gradient tensor of shape (..., M, K) or (batch, M, K)
        coarse_dim_perc: Percentage of dimensions to sample
        rank: Rank for truncated SVD
        eps: Epsilon for stability

    Returns:
        Preconditioned gradient of same shape as G
    """
    original_shape = G.shape
    original_dtype = G.dtype
    batch_size = G.shape[0] if G.ndim == 3 else 1

    # Handle both batched and unbatched inputs
    if G.ndim == 2:
        G = G.unsqueeze(0)

    # Process each item in batch
    results = []
    for i in range(G.shape[0]):
        g = G[i]  # Shape: (M, K)
        n, d = g.shape

        # Convert to float32 for SVD (doesn't support bfloat16)
        g_f32 = g.float()

        # Coarse dimension sampling (KEY DIFFERENCE from polar methods)
        k = min(n, math.ceil(coarse_dim_perc * n) + 1)
        idx = torch.randperm(n, device=g.device)[:k]

        # Sample subset of gradients
        gsub = g_f32.index_select(0, idx)  # (k, d)

        # Compute Gram matrix P = G @ G^T (on sampled dimensions)
        P = gsub @ gsub.T  # (k, k)

        # Truncated SVD on Gram matrix
        if rank < k:
            U, S, _ = torch.svd_lowrank(P, q=rank)
        else:
            U, S, _ = torch.linalg.svd(P, full_matrices=False)

        # Eigenvalue processing: sqrt and clamp
        S = S.abs().sqrt_().clamp(min=eps)

        # Get last singular value for threshold
        sigma_last = S[-1]

        # Use all but last singular vector
        U_m1 = U[:, :-1]  # (k, r-1)

        # Compute correction term
        Ut = U_m1.T  # (r-1, k)
        a = Ut @ gsub  # (r-1, d)

        # Scale by (1/S[:-1] - 1/sigma_last)
        inv = (1.0 / S[:-1]) - (1.0 / sigma_last)  # (r-1,)
        a = a * inv.unsqueeze(1)  # (r-1, d)

        c = U_m1 @ a  # (k, d)

        # Final update for sampled dimensions
        dH = (-gsub / sigma_last) - c  # (k, d)

        # Convert back to original dtype
        dH = dH.to(original_dtype)

        # Create full update tensor (sparse update)
        update = torch.zeros_like(g)
        update.index_add_(0, idx, dH)

        results.append(update)

    # Stack results and reshape
    result = torch.stack(results)
    if batch_size == 1:
        result = result.squeeze(0)

    return result.view(original_shape)
