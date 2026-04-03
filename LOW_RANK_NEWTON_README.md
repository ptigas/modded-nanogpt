# Low-Rank Newton Methods Integration

Integration of ideas from ["A Multilevel Low-Rank Newton Method with Super-Linear Convergence Rate and Its Application to Non-Convex Problems"](https://arxiv.org/abs/2305.08742) by Tsipinakis, Tigkas, and Parpas (2023).

## Key Benefits

1. **Better Saddle Point Escape**: Handles negative eigenvalues by taking absolute values
2. **Robust to Small Eigenvalues**: Thresholds small eigenvalues to avoid numerical issues
3. **Faster Convergence**: Super-linear convergence rate in certain regions
4. **Compile-Friendly**: All methods are `@torch.compile` compatible

## Quick Start

### Test the Implementation

First, verify everything works:

```bash
uv run test_low_rank_newton.py
```

### Enable Truncated SVD Method

In `train_gpt.py`, modify the `optimizer2` initialization:

```python
optimizer2 = NorMuon(
    hidden_matrix_params + gate_params,
    lr=0.03,
    momentum=0.95,
    beta2=0.95,
    weight_decay=0.0,
    use_truncated_svd=True,   # Enable the new method
    newton_rank=128,           # SVD rank
    newton_nu=1e-6,            # Eigenvalue threshold
    hybrid_blend=0.0,          # 0.0 = pure SVD
)
```

## Configuration Options

### `use_truncated_svd` (bool, default=False)
- `False`: Use original `polar_express` method
- `True`: Use truncated SVD-based preconditioning

### `newton_rank` (int, default=128)
- Number of singular values to keep in truncated SVD
- **Lower values** (64-128): Faster, less accurate, better for saddle escape
- **Higher values** (256-512): Slower, more accurate, closer to full Newton
- **Rule of thumb**: Start with `dim // 4` to `dim // 2`

### `newton_nu` (float, default=1e-6)
- Minimum eigenvalue threshold
- **Smaller values** (1e-8 to 1e-10): More aggressive thresholding
- **Larger values** (1e-4 to 1e-3): More conservative, better for saddle escape
- **Adaptive mode**: Automatically adjusts based on gradient magnitude

### `hybrid_blend` (float, default=0.0)
- Blend factor between methods
- `0.0`: Pure truncated SVD (from paper)
- `0.5`: 50/50 blend of polar and SVD
- `1.0`: Pure polar decomposition (original method)

## Experimental Configurations

### Experiment 1: Pure Truncated SVD (Recommended Start)
```python
use_truncated_svd=True
newton_rank=128
newton_nu=1e-6
hybrid_blend=0.0
```
**Expected**: Better saddle escape, similar or slightly slower per-step time

### Experiment 2: High-Rank SVD (More Accurate)
```python
use_truncated_svd=True
newton_rank=256
newton_nu=1e-8
hybrid_blend=0.0
```
**Expected**: Closer to full Newton, better convergence, slower per-step

### Experiment 3: Low-Rank SVD (Saddle Escape Focus)
```python
use_truncated_svd=True
newton_rank=64
newton_nu=1e-4  # More aggressive threshold
hybrid_blend=0.0
```
**Expected**: Fastest saddle escape, potentially less stable

### Experiment 4: Hybrid Method (Safe Start)
```python
use_truncated_svd=True
newton_rank=128
newton_nu=1e-6
hybrid_blend=0.3  # 30% polar, 70% SVD
```
**Expected**: Balanced performance, good stability

### Experiment 5: Adaptive Hybrid (Advanced)
Start with hybrid, then switch to pure SVD after warmup:
```python
# Steps 0-1000: hybrid_blend=0.5
# Steps 1000+: hybrid_blend=0.0
```

## Performance Expectations

### Computational Cost
- **polar_express**: ~5 iterations of matrix-matrix ops
- **truncated_svd (rank=128)**: 1 SVD + low-rank update
- **Overhead**: 5-15% slower per step, but potentially fewer steps needed

### Memory Usage
- Similar to polar_express (no significant increase)
- SVD uses temporary buffers of size `(batch, rank, dim)`

### Convergence
Based on the paper's results:
- **Convex regions**: Super-linear convergence (faster than linear)
- **Saddle points**: Much faster escape than first-order methods
- **Overall**: Potentially 10-30% fewer steps to convergence

## Monitoring and Debugging

### Check if Method is Active
Add logging after optimizer creation:
```python
print(f"NorMuon using truncated SVD: {optimizer2.use_truncated_svd}")
print(f"SVD rank: {optimizer2.newton_rank}, nu: {optimizer2.newton_nu}")
```

### Monitor Gradient Norms
```python
# After backward, before optimizer.step()
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Step {step}, grad_norm: {grad_norm:.4f}")
```

### Detect Plateau/Saddle Regions
Use the included utility:
```python
from low_rank_newton import detect_saddle_region

grad_history = []
# In training loop:
grad_history.append(grad_norm.item())
if detect_saddle_region(grad_history, window=50):
    print(f"Possible saddle point detected at step {step}")
```

## Troubleshooting

### Issue: NaN/Inf in gradients
**Solution**: Increase `newton_nu` to 1e-4 or higher
```python
newton_nu=1e-4  # More conservative threshold
```

### Issue: Slower convergence
**Solution**: Try hybrid blend or increase rank
```python
hybrid_blend=0.3  # Add some polar decomposition
# OR
newton_rank=256   # Use more singular values
```

### Issue: Out of memory
**Solution**: Decrease rank
```python
newton_rank=64    # Reduce SVD rank
```

### Issue: Compilation errors
**Solution**: All methods are compile-friendly, but if issues arise:
```python
# The methods are already @torch.compile decorated
# If you get recompilation warnings, they're expected on first run
```

## Theory: Why This Works

### Eigenvalue Modification
The paper's key insight is modifying the Hessian eigenvalues:
```
g(σ) = max(|σ|, ν)
```

This transformation:
1. **Negative eigenvalues** → absolute value (prevents moving toward saddle)
2. **Small eigenvalues** → threshold (avoids division by near-zero)
3. **Large eigenvalues** → unchanged (keeps important curvature)

### Comparison to Polar Express
- **Polar Express**: Finds nearest orthogonal matrix (preserve directions)
- **Truncated SVD**: Finds nearest low-rank approximation + eigenvalue modification
- **Hybrid**: Blends both properties

### When to Use Each
- **Polar Express**: Stable, proven, good general performance
- **Pure SVD**: Better saddle escape, potentially faster convergence
- **Hybrid**: Best of both worlds, safer experimentation

## Advanced: Custom Schedules

### Adaptive Rank Schedule
```python
def get_newton_rank(step):
    if step < 500:
        return 64   # Low rank during warmup
    elif step < 1500:
        return 128  # Medium rank during main training
    else:
        return 256  # High rank for fine-tuning

# In training loop, before optimizer.step():
optimizer2.newton_rank = get_newton_rank(step)
```

### Adaptive Threshold Schedule
```python
def get_newton_nu(step, val_loss):
    # Increase threshold if loss plateaus
    if val_loss_improving:
        return 1e-6   # Standard threshold
    else:
        return 1e-4   # Higher threshold for saddle escape
```

## Citation

If this helps your research, consider citing:
```bibtex
@article{tsipinakis2023multilevel,
  title={A Multilevel Low-Rank Newton Method with Super-Linear Convergence Rate and Its Application to Non-Convex Problems},
  author={Tsipinakis, Nick and Tigkas, Panagiotis and Parpas, Panos},
  journal={arXiv preprint arXiv:2305.08742},
  year={2023}
}
```

## Further Reading

- [Original Paper](https://arxiv.org/abs/2305.08742)
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)
- [Second-Order Methods Survey](https://arxiv.org/abs/1912.08957)
