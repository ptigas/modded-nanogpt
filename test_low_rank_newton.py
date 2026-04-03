"""
Test script for Low-Rank Newton methods.
Verifies torch.compile compatibility and basic functionality.
"""

import torch
import time
from low_rank_newton import (
    truncated_svd_precondition,
    adaptive_truncated_svd_precondition,
    hybrid_polar_newton,
    simplified_polar,
    detect_saddle_region
)


def test_basic_functionality():
    """Test that all methods produce valid output"""
    print("\n=== Testing Basic Functionality ===")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with various shapes
    shapes = [
        (256, 512),   # Tall matrix
        (512, 256),   # Wide matrix
        (384, 384),   # Square matrix
    ]

    for shape in shapes:
        print(f"\nTesting shape {shape}...")
        G = torch.randn(shape, device=device, dtype=torch.bfloat16)

        # Test truncated SVD
        result_svd = truncated_svd_precondition(G, rank=64, nu=1e-6)
        assert result_svd.shape == G.shape, f"Shape mismatch: {result_svd.shape} != {G.shape}"
        assert not torch.isnan(result_svd).any(), "NaN detected in SVD output"
        print(f"  ✓ Truncated SVD: shape={result_svd.shape}, norm={result_svd.norm():.4f}")

        # Test adaptive SVD
        result_adaptive = adaptive_truncated_svd_precondition(G, rank=64)
        assert result_adaptive.shape == G.shape
        assert not torch.isnan(result_adaptive).any(), "NaN detected in adaptive output"
        print(f"  ✓ Adaptive SVD: shape={result_adaptive.shape}, norm={result_adaptive.norm():.4f}")

        # Test hybrid
        result_hybrid = hybrid_polar_newton(G, rank=64, blend_factor=0.5)
        assert result_hybrid.shape == G.shape
        assert not torch.isnan(result_hybrid).any(), "NaN detected in hybrid output"
        print(f"  ✓ Hybrid: shape={result_hybrid.shape}, norm={result_hybrid.norm():.4f}")

        # Test simplified polar
        result_polar = simplified_polar(G, num_iters=3)
        assert result_polar.shape == G.shape
        assert not torch.isnan(result_polar).any(), "NaN detected in polar output"
        print(f"  ✓ Simplified Polar: shape={result_polar.shape}, norm={result_polar.norm():.4f}")

    print("\n✓ All basic functionality tests passed!")


def test_compile_compatibility():
    """Test torch.compile compatibility"""
    print("\n=== Testing torch.compile Compatibility ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping compile test on CPU (compile is CUDA-optimized)")
        return

    torch.manual_seed(42)
    G = torch.randn(384, 768, device=device, dtype=torch.bfloat16)

    # The functions are already decorated with @torch.compile
    # Test that they run without recompilation errors
    print("Testing compiled truncated_svd_precondition...")
    result1 = truncated_svd_precondition(G, rank=128)
    result2 = truncated_svd_precondition(G, rank=128)  # Should use cached version
    print(f"  ✓ Compiled successfully, output norm={result1.norm():.4f}")

    print("\n✓ torch.compile compatibility confirmed!")


def test_eigenvalue_handling():
    """Test that negative and small eigenvalues are handled correctly"""
    print("\n=== Testing Eigenvalue Handling ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Create a matrix with known eigenvalue structure
    # Use a symmetric matrix for clear eigenvalues
    n = 256
    A = torch.randn(n, n, device=device, dtype=torch.float32)
    A = (A + A.T) / 2  # Make symmetric

    # Add some negative eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues[:50] = -eigenvalues[:50].abs()  # Make first 50 negative
    eigenvalues[50:100] = 1e-8  # Make next 50 very small
    A_modified = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    print(f"Input matrix eigenvalues: min={eigenvalues.min():.2e}, max={eigenvalues.max():.2e}")
    print(f"  Negative eigenvalues: {(eigenvalues < 0).sum().item()}")
    print(f"  Small eigenvalues (< 1e-6): {(eigenvalues.abs() < 1e-6).sum().item()}")

    # Apply truncated SVD preconditioning
    nu = 1e-6
    result = truncated_svd_precondition(A_modified.bfloat16(), rank=128, nu=nu)

    print(f"\nOutput matrix: shape={result.shape}, norm={result.norm():.4f}")
    print(f"  Contains NaN: {torch.isnan(result).any()}")
    print(f"  Contains Inf: {torch.isinf(result).any()}")

    # The preconditioned result should not have NaN/Inf
    assert not torch.isnan(result).any(), "NaN detected in output"
    assert not torch.isinf(result).any(), "Inf detected in output"

    print("\n✓ Eigenvalue handling test passed!")


def benchmark_methods():
    """Benchmark different methods"""
    print("\n=== Benchmarking Methods ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Skipping benchmark on CPU (use CUDA for accurate timings)")
        return

    torch.manual_seed(42)
    shape = (768, 3072)  # Typical GPT-2 MLP shape
    G = torch.randn(shape, device=device, dtype=torch.bfloat16)

    methods = {
        "Truncated SVD (rank=64)": lambda: truncated_svd_precondition(G, rank=64),
        "Truncated SVD (rank=128)": lambda: truncated_svd_precondition(G, rank=128),
        "Truncated SVD (rank=256)": lambda: truncated_svd_precondition(G, rank=256),
        "Adaptive SVD": lambda: adaptive_truncated_svd_precondition(G, rank=128),
        "Hybrid (50/50)": lambda: hybrid_polar_newton(G, rank=128, blend_factor=0.5),
        "Simplified Polar": lambda: simplified_polar(G, num_iters=3),
    }

    # Warmup
    for _ in range(10):
        for method_fn in methods.values():
            _ = method_fn()
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 100
    results = {}

    for name, method_fn in methods.items():
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(num_iters):
            _ = method_fn()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_time = elapsed / num_iters * 1000  # Convert to ms
        results[name] = avg_time

    print(f"\nBenchmark results (averaged over {num_iters} iterations):")
    print(f"Input shape: {shape}")
    for name, avg_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:30s}: {avg_time:6.3f} ms")

    print("\n✓ Benchmark completed!")


def test_saddle_detection():
    """Test saddle point detection utility"""
    print("\n=== Testing Saddle Detection ===")

    # Simulate gradient norms that plateau
    grad_norms_stuck = [1.0, 0.99, 1.01, 0.98, 1.02, 0.99, 1.0, 1.01, 0.99, 1.0] * 5
    is_stuck = detect_saddle_region(grad_norms_stuck, window=20, threshold=1e-3)
    print(f"Stuck gradient pattern: detected={is_stuck} (expected: True)")
    assert is_stuck, "Failed to detect stuck gradients"

    # Simulate healthy gradient norms
    grad_norms_healthy = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    is_healthy = detect_saddle_region(grad_norms_healthy, window=10, threshold=1e-3)
    print(f"Healthy gradient pattern: detected={is_healthy} (expected: False)")
    assert not is_healthy, "False positive on healthy gradients"

    print("\n✓ Saddle detection test passed!")


def test_batch_dimension():
    """Test with batch dimension (for multi-layer processing)"""
    print("\n=== Testing Batch Dimension ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Shape: (batch, M, K)
    G_batched = torch.randn(4, 256, 512, device=device, dtype=torch.bfloat16)

    result = truncated_svd_precondition(G_batched, rank=64)
    assert result.shape == G_batched.shape, f"Shape mismatch: {result.shape} != {G_batched.shape}"
    assert not torch.isnan(result).any(), "NaN detected in batched output"

    print(f"Batched processing: input={G_batched.shape}, output={result.shape}")
    print(f"  Output norm: {result.norm():.4f}")
    print("\n✓ Batch dimension test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Low-Rank Newton Methods - Test Suite")
    print("=" * 60)

    # Run all tests
    test_basic_functionality()
    test_compile_compatibility()
    test_eigenvalue_handling()
    test_saddle_detection()
    test_batch_dimension()
    benchmark_methods()

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now use these methods in train_gpt.py")
    print("Set use_truncated_svd=True in NorMuon to enable.")
