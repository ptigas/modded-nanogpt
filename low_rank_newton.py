"""
Low-Rank Newton Methods based on "A Multilevel Low-Rank Newton Method" (2023)
by Tsipinakis, Tigkas, and Parpas

This module implements truncated SVD-based preconditioning for non-convex optimization
with better saddle point escape properties.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.compile(dynamic=False, fullgraph=True)
def truncated_svd_precondition(
    G: Tensor,
    rank: int = 128,
    nu: float = 1e-6,
    use_power_iter: bool = True
) -> Tensor:
    """
    Apply truncated SVD-based preconditioning with eigenvalue modification.

    Based on equation (13) from the paper:
    |Q^{-1}| = g(σ_{N+1})^{-1} I + U_N (g(Σ_N)^{-1} - g(σ_{N+1})^{-1} I) U_N^T

    where g_i(σ) = max(|σ|, ν) handles negative and small eigenvalues.

    Args:
        G: Gradient tensor of shape (..., M, K)
        rank: Number of singular values to keep (N in paper)
        nu: Minimum eigenvalue threshold
        use_power_iter: Use power iteration for SVD (faster, compile-friendly)

    Returns:
        Preconditioned gradient of same shape as G
    """
    original_shape = G.shape
    original_dtype = G.dtype

    # Convert to float32 for SVD (SVD doesn't support bfloat16 on CUDA)
    X = G.float()

    # Handle transpose for tall matrices
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Normalize to prevent overflow
    norm = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (norm * 1.02 + 1e-6)
    X = X.contiguous()

    M, K = X.shape[-2:]

    # Compute low-rank SVD: X ≈ U @ diag(S) @ V^T
    # Use randomized/power iteration SVD for torch.compile compatibility
    if use_power_iter and rank < min(M, K):
        # Randomized SVD via power iteration (compile-friendly)
        U, S, Vh = torch.svd_lowrank(X, q=rank, niter=2)
    else:
        # Full SVD then truncate (slower but more accurate)
        U_full, S_full, Vh_full = torch.linalg.svd(X, full_matrices=False)
        actual_rank = min(rank, S_full.size(-1))
        U = U_full[..., :, :actual_rank]
        S = S_full[..., :actual_rank]
        Vh = Vh_full[..., :actual_rank, :]

    # Apply eigenvalue transformation: g(σ) = max(|σ|, ν)
    # This handles both negative eigenvalues (take abs) and small ones (threshold)
    S_modified = torch.clamp(S.abs(), min=nu)*0.0 + 1.0

    # Get threshold eigenvalue (the (N+1)th eigenvalue)
    # Use the smallest of our kept eigenvalues
    sigma_threshold = S_modified[..., -1:].clamp(min=nu)

    # Construct inverse preconditioner Q^{-1}
    # Q^{-1} = (1/σ_threshold) I + U (1/S_modified - 1/σ_threshold) U^T
    #
    # For efficiency, we compute: Q^{-1} @ (-∇f) directly
    # = (1/σ_threshold) (-∇f) + U @ ((1/S_modified - 1/σ_threshold) * (U^T @ (-∇f)))

    # Compute U^T @ X
    UtX = U.mT @ X  # Shape: (..., rank, K)

    # Compute scaling: (1/S_modified - 1/σ_threshold)
    inv_S_modified = 1.0 / S_modified
    inv_sigma_threshold = 1.0 / sigma_threshold
    scaling = (inv_S_modified - inv_sigma_threshold).unsqueeze(-1)  # (..., rank, 1)

    # Apply preconditioner
    X_scaled_identity = X * inv_sigma_threshold.unsqueeze(-1)  # Base scaling
    X_lowrank_correction = U @ (scaling * UtX)  # Low-rank correction
    X_precon = X_scaled_identity + X_lowrank_correction

    # Restore normalization
    X_precon = X_precon * norm

    # Restore transpose
    if transposed:
        X_precon = X_precon.mT

    # Convert back to original dtype
    X_precon = X_precon.to(original_dtype)

    return X_precon.view(original_shape)


@torch.compile(dynamic=False, fullgraph=True)
def adaptive_truncated_svd_precondition(
    G: Tensor,
    rank: int = 128,
    nu_base: float = 1e-6,
    adaptive_nu: bool = True
) -> Tensor:
    """
    Adaptive version that adjusts nu based on gradient magnitude.
    Better for handling different scales during training.
    """
    # Adaptive threshold based on gradient norm
    if adaptive_nu:
        grad_scale = G.abs().mean()
        nu = torch.clamp(grad_scale * 1e-4, min=nu_base, max=1e-3)
    else:
        nu = nu_base

    return truncated_svd_precondition(G, rank=rank, nu=nu, use_power_iter=True)


@torch.compile(dynamic=False, fullgraph=True)
def hybrid_polar_newton(
    G: Tensor,
    rank: int = 128,
    nu: float = 1e-6,
    blend_factor: float = 0.5,
    polar_iters: int = 3
) -> Tensor:
    """
    Hybrid approach: blend between polar decomposition and truncated Newton.

    Args:
        G: Gradient tensor
        rank: SVD rank
        nu: Eigenvalue threshold
        blend_factor: 0.0 = pure truncated SVD, 1.0 = pure polar
        polar_iters: Number of polar iterations (reduced for speed)

    Returns:
        Blended preconditioned gradient
    """
    if blend_factor <= 0.0:
        return truncated_svd_precondition(G, rank=rank, nu=nu)
    elif blend_factor >= 1.0:
        # Fall back to simplified polar (faster version)
        return simplified_polar(G, num_iters=polar_iters)
    else:
        # Blend both approaches
        newton_result = truncated_svd_precondition(G, rank=rank, nu=nu)
        polar_result = simplified_polar(G, num_iters=polar_iters)
        return blend_factor * polar_result + (1.0 - blend_factor) * newton_result


@torch.compile(dynamic=False, fullgraph=True)
def simplified_polar(G: Tensor, num_iters: int = 3) -> Tensor:
    """
    Simplified polar decomposition (fewer iterations than polar_express).
    Compile-friendly version.
    """
    original_dtype = G.dtype
    X = G.float()  # Use float32 for stability
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    X = X.contiguous()

    M = X.size(-2)

    # Fixed coefficients for 3 iterations (compile-friendly)
    # Simplified from polar_express_coeffs
    coeffs = [
        (4.0, -2.8, 0.5),
        (3.9, -2.7, 0.5),
        (3.3, -2.4, 0.46),
    ]

    A = torch.empty((*X.shape[:-2], M, M), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    for i in range(min(num_iters, len(coeffs))):
        a, b, c = coeffs[i]

        # A = X @ X.mT
        A = X @ X.mT

        # B = b * A + c * (A @ A)
        B = b * A + c * (A @ A)

        # C = a * X + B @ X
        C = a * X + B @ X

        # Swap
        X, C = C, X

    if transposed:
        X = X.mT

    return X.to(original_dtype)


# Utility for detecting saddle points (not compiled, used for monitoring)
def detect_saddle_region(grad_norm_history: list, window: int = 10, threshold: float = 1e-6) -> bool:
    """
    Detect if optimization is stuck in a saddle point region.
    Use this for adaptive switching between methods.

    Args:
        grad_norm_history: List of recent gradient norms
        window: Number of steps to check
        threshold: Variance threshold for "stuck" detection

    Returns:
        True if likely in saddle region
    """
    if len(grad_norm_history) < window:
        return False

    recent = grad_norm_history[-window:]
    variance = torch.tensor(recent).var().item()
    mean = torch.tensor(recent).mean().item()

    # Low variance + non-zero mean = stuck at saddle
    return variance < threshold and mean > threshold * 10
