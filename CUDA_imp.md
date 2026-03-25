# CUDA Integration Strategy for rust_ann

## Hardware Profile

| Spec | Value |
|------|-------|
| GPU | RTX 3050 Laptop (Ampere SM86) |
| CUDA Cores | 2048 |
| VRAM | 4 GB GDDR6 |
| FP32 | ~5 TFLOPS |
| FP64 | ~156 GFLOPS (1:32 ratio — unusable) |
| Driver | 590.48.01 |
| CUDA Support | 13.1 |

---

## Tape Architecture Constraint: param_count Boundary

The Tape splits its tensor storage into two zones:

```
tensors[0 .. param_count]       ← Frozen parameters (weights, biases, embeddings)
                                   Created before tape.freeze_params()
                                   Persist across tape.reset() calls
                                   requires_grad = true

tensors[param_count .. len]     ← Ephemeral intermediates (activations, loss)
                                   Created during forward/backward
                                   Destroyed by tape.reset() each batch
                                   requires_grad = false (typically)
```

**This boundary is load-bearing for GPU migration:**

1. **Optimizer kernels (SGD/Adam) must only iterate `[0..param_count]`.**
   Bug discovered: Adam's `step()` originally iterated all `tape.tensors`, causing
   index-out-of-bounds because Adam's `m`/`v` vectors are sized to `param_count`.
   Fix: `tape.tensors[..tape.param_count].iter_mut()`.

2. **GPU memory for parameters is persistent; for intermediates it is per-batch.**
   - Parameter `CudaSlice` allocations live for the entire training run.
   - Intermediate `CudaSlice` allocations are freed on `tape.reset()`.

3. **Adam state (`m`, `v` vectors) must also live on GPU** to avoid per-step D→H→D
   transfers. These are parameter-sized, allocated once at `Adam::new()`.

---

## Current Optimizer: Adam

Added to `tensor.rs`, replacing SGD for the ViT pipeline:

```rust
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,       // 0.9
    pub beta2: f64,       // 0.999
    pub eps: f64,         // 1e-8
    pub t: usize,         // timestep (incremented per step)
    pub m: Vec<Vec<f64>>, // first moment per param tensor
    pub v: Vec<Vec<f64>>, // second moment per param tensor
}
```

**GPU implication:** Adam requires 3x the parameter memory (params + m + v) vs SGD's 1x.
The `m` and `v` vectors must be mirrored as `CudaSlice` on GPU for the adam_step kernel
to run without host round-trips.

---

## Critical Decision: f64 → f32

The RTX 3050 has a **1:32 FP64:FP32 ratio**. Running f64 on this GPU is slower than CPU.
All tensor data must migrate from `f64` to `f32`. This also doubles CPU SIMD throughput (8×f32 vs 4×f64 per AVX register).

For MNIST-scale training, f32 precision is more than sufficient.

**Migration scope:**
- `Tensor.data: Vec<f64>` → `Vec<f32>`
- `Tensor.grad: Vec<f64>` → `Vec<f32>`
- All arithmetic literals (`0.0` stays, explicit `0.0_f64` → `0.0_f32`)
- `raw_matmul`, `raw_transpose`, `positional_encoding` signatures
- Learning rate, loss accumulation — all f32

---

## Recommended Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Rust ↔ CUDA | `cudarc` crate | Safe typed bindings, maintained, supports cuBLAS + cuDNN + custom kernels |
| Matrix multiply | cuBLAS `cublasSgemm` | Vendor-optimized, ~50-100x over triple-loop CPU |
| Convolution | cuDNN `cudnnConvolutionForward` | Vendor-optimized, handles im2col internally |
| Element-wise ops | Custom `.cu` kernels via `cudarc::nvrtc` | Trivial to write, compiled at build time |
| Fallback | Existing CPU code behind `#[cfg(feature)]` | Graceful degradation when no GPU |

### Alternatives Considered

| Option | Why Not |
|--------|---------|
| `rust-cuda` / Rust-GPU | Nightly-only, experimental, poor library support |
| Raw `cuda-sys` FFI | Massive unsafe boilerplate for no benefit over cudarc |
| `wgpu` compute shaders | No cuBLAS access, poor matmul performance |
| `tch-rs` (libtorch) | Replaces entire framework — defeats the purpose |

---

## Architecture

### Principle: GPU as Accelerator Behind Tape

The Tape owns all tensors. GPU memory is an optional mirror. No change to the ownership/autograd model.

```
Tape
 ├── tensors: Vec<Tensor>
 │    ├── data: Vec<f32>              ← CPU storage (source of truth when dirty_device)
 │    ├── grad: Vec<f32>              ← CPU grads
 │    ├── d_data: Option<CudaSlice>   ← GPU mirror (source of truth when dirty_host)
 │    ├── d_grad: Option<CudaSlice>   ← GPU grads
 │    ├── dirty_host: bool            ← CPU data is stale
 │    └── dirty_device: bool          ← GPU data is stale
 │
 └── gpu: Option<GpuDevice>
      ├── dev: Arc<CudaDevice>
      └── blas: CudaBlas
```

### Memory Transfer Rules

1. **Lazy upload**: Data stays on CPU until a GPU op requests it
2. **Lazy download**: Results stay on GPU until CPU reads them (loss printing, argmax)
3. **Dirty flags** prevent redundant H↔D copies
4. **During forward+backward**, data stays resident on GPU — no round-trips
5. **Only scalar loss** gets downloaded per batch for logging

### Feature Gating

```toml
# Cargo.toml
[features]
default = ["cuda"]
cuda = ["cudarc"]
cpu-only = []

[dependencies]
cudarc = { version = "0.12", features = ["cublas", "cudnn"], optional = true }
```

All GPU code paths are behind `#[cfg(feature = "cuda")]`. The CPU fallback is always available.

---

## Op Offload Priority

### P0 — Matrix Multiply (>90% of runtime)

| What | CPU Current | GPU Target |
|------|-------------|------------|
| Forward `matmul` | Triple loop O(m·k·n) | `cublasSgemm` |
| Backward `dA = dC @ B^T` | Triple loop | `cublasSgemm` with transpose flag |
| Backward `dB = A^T @ dC` | Triple loop | `cublasSgemm` with transpose flag |

This covers `Linear.forward`, all attention projections, and the entire transformer FFN.
**Expected speedup: 50-100x per matmul.**

### P1 — Convolution

| What | CPU Current | GPU Target |
|------|-------------|------------|
| Forward `conv2d` | 6-nested loop | `cudnnConvolutionForward` |
| Backward input grad | 6-nested loop | `cudnnConvolutionBackwardData` |
| Backward kernel grad | 6-nested loop | `cudnnConvolutionBackwardFilter` |

**Expected speedup: 30-80x.**

### P2 — Element-wise Ops (custom CUDA kernels)

```
relu          : y[i] = max(0, x[i])
relu_backward : dx[i] = (x[i] > 0) ? dy[i] : 0
add           : z[i] = a[i] + b[i]
add_broadcast : z[row*n+col] = a[row*n+col] + bias[col]
scalar_mul    : y[i] = x[i] * scalar
sgd_step      : param[i] -= lr * grad[i]; grad[i] = 0
adam_step     : m[i] = β1*m[i] + (1-β1)*g[i]
                v[i] = β2*v[i] + (1-β2)*g[i]*g[i]
                param[i] -= lr * (m[i]/(1-β1^t)) / (sqrt(v[i]/(1-β2^t)) + eps)
                g[i] = 0
```

Each is a one-line kernel (adam_step is ~6 lines). Compile via `cudarc::nvrtc` at build time.
**Expected speedup: 5-20x.** Main benefit is avoiding D→H→D round-trips between GPU matmuls.

**Critical:** Both `sgd_step` and `adam_step` kernels must only launch over parameter
tensors `[0..param_count]`, not ephemeral intermediates. The kernel grid is sized to
param element count, and `m`/`v` device buffers only cover param tensors.

### P3 — Reduction / Specialized Ops

```
softmax       : shared-memory reduction per row
layernorm     : mean + variance reduction per row, then element-wise
mse_loss      : parallel diff-square + tree reduction
cross_entropy : log-softmax + nll in one kernel
max_pool2d    : strided grid launch
embedding     : gather kernel (index into table)
```

**Expected speedup: 5-30x.** These are memory-bound; GPU wins from bandwidth, not compute.

---

## Memory Budget (4 GB VRAM)

With Adam optimizer, each parameter tensor needs 3× storage: params + first moment (m) + second moment (v).

| Model | Params | Params (f32) | Adam m+v (f32) | Peak Intermediate | Total |
|-------|--------|-------------|----------------|-------------------|-------|
| ANN (784→128→10) | 101K | 0.4 MB | 0.8 MB | ~5 MB | **~6.2 MB** |
| CNN (conv+pool+FC) | 5K | 0.02 MB | 0.04 MB | ~10 MB | **~10.1 MB** |
| Transformer (d=64, 2L) | ~200K | 0.8 MB | 1.6 MB | ~30 MB | **~32.4 MB** |
| ViT (d=32, 2L) | 18K | 0.07 MB | 0.14 MB | ~20 MB | **~20.2 MB** |

All models fit comfortably. Adam's 3× param overhead is negligible at this scale.
Even with batch size 128, peak usage stays under 200 MB.

---

## Implementation Phases

### Phase 1: Foundation (biggest impact)

```
1. Install CUDA toolkit:
   $ sudo pacman -S cuda cudnn

2. Add .cargo/config.toml for CPU optimization:
   [build]
   rustflags = ["-C", "target-cpu=native"]

3. Global f64 → f32 migration across all .rs files

4. Add cudarc to Cargo.toml

5. Create src/gpu.rs:
   - GpuDevice struct (CudaDevice + CudaBlas)
   - matmul_f32() wrapping cublasSgemm
   - ensure_on_device() / ensure_on_host() transfer helpers

6. Add Option<GpuDevice> to Tape

7. GPU-accelerate Tape::matmul() forward path
   - Upload inputs lazily
   - Store result on GPU (no download)
   - CPU fallback when gpu is None

8. GPU-accelerate matmul backward in Tape::backward()
   - dA = dC @ B^T via cublasSgemm with CUBLAS_OP_T
   - dB = A^T @ dC via cublasSgemm with CUBLAS_OP_T
```

**Result: ANN training 20-50x faster. CNN linear layer 20-50x faster.**

### Phase 2: Element-wise Kernels + Adam on GPU

```
1. Write CUDA kernels as inline strings (compiled via nvrtc):
   - relu_forward / relu_backward
   - add_elementwise
   - add_broadcast
   - scalar_mul
   - sgd_step (update params in-place on GPU)
   - adam_step (update params + m + v in-place on GPU)

2. Register kernels at GpuDevice creation time

3. Route Tape ops through GPU kernels when device is available

4. Adam GPU state:
   - Allocate d_m, d_v as CudaSlice<f32> per param tensor at Adam::new()
   - adam_step kernel reads grad, updates m, v, param — all on device
   - Scalars (lr, beta1, beta2, eps, t) passed as kernel args
   - CRITICAL: only launch over tensors[0..param_count]

5. sgd_step/adam_step on GPU eliminates per-batch D→H→D for param updates
```

**Result: Entire forward+backward+update stays on GPU. Only loss scalar transfers.**

### Phase 3: Convolution via cuDNN

```
1. Initialize cudnnHandle via cudarc::cudnn

2. Create persistent tensor/filter/conv descriptors for each Conv2D layer

3. GPU conv2d forward:
   - cudnnConvolutionForward with IMPLICIT_PRECOMP_GEMM algo

4. GPU conv2d backward:
   - cudnnConvolutionBackwardData (input gradient)
   - cudnnConvolutionBackwardFilter (kernel gradient)

5. GPU max_pool2d:
   - cudnnPoolingForward / cudnnPoolingBackward
```

**Result: CNN training fully on GPU. 30-80x speedup on conv layers.**

### Phase 4: Specialized Kernels

```
1. softmax: row-wise shared-memory reduction kernel
2. layernorm: two-pass (mean, variance) with shared memory
3. mse_loss / cross_entropy: parallel reduction
4. embedding: gather kernel with gradient scatter (atomicAdd)
5. transpose: tiled shared-memory transpose kernel
```

**Result: Transformer/ViT training fully on GPU end-to-end.**

---

## Quick CPU Wins (No CUDA Required)

These are free and stack with future GPU work:

| Change | Speedup | Effort |
|--------|---------|--------|
| f64 → f32 | ~2x | Find-replace + fix types |
| Loop reorder in `raw_matmul`: `i,j,p` → `i,p,j` | 3-5x | 2 lines |
| `-C target-cpu=native` in .cargo/config.toml | 1.5-2x | 1 file |
| Batch CNN training (currently batch_size=1) | 5-10x | Moderate refactor |

Combined CPU-only improvement: **~10-30x before touching CUDA.**

---

## File Structure After Integration

```
src/
├── lib.rs              ← add pub mod gpu;
├── gpu.rs              ← NEW: GpuDevice, transfer helpers, kernel registry
├── kernels/            ← NEW: .cu files for custom ops
│   ├── elementwise.cu  ←   relu, add, scalar_mul, sgd_step
│   ├── reduction.cu    ←   softmax, layernorm, loss functions
│   └── pooling.cu      ←   max_pool2d
├── tensor.rs           ← extend Tensor with d_data/d_grad, GPU dispatch
├── layers.rs           ← unchanged (ops route through Tape)
├── main.rs             ← init GpuDevice, pass to Tape
├── mnist.rs            ← unchanged
└── transformer_train.rs ← unchanged

.cargo/
└── config.toml         ← NEW: target-cpu=native

Cargo.toml              ← add cudarc, feature flags
```

---

## Risk & Mitigation

| Risk | Mitigation |
|------|------------|
| cudarc API changes | Pin version, `cudarc = "=0.12.x"` |
| CUDA toolkit missing on CI | `cpu-only` feature flag, test both paths |
| 4 GB VRAM pressure on bigger models | Track allocations, free intermediates in `tape.reset()` |
| cuBLAS column-major vs row-major | Trick: compute `B^T @ A^T` to get `(A@B)^T` in col-major = row-major |
| Debugging GPU numerics | Compare GPU vs CPU output for first N batches, assert tolerance <1e-5 |
| Optimizer iterates past param_count | **Proven bug.** Always slice `[..tape.param_count]` in optimizer step. GPU kernels must be grid-sized to param element counts only |
| Adam m/v out of sync with tape | `Adam::new()` must be called after `tape.freeze_params()`. If model structure changes, Adam must be re-initialized |
