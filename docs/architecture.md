# Autograd Engine Architecture

## Overview

The core of this framework is a **tape-based reverse-mode automatic differentiation** engine implemented in `tensor.rs`. Every forward operation records itself on a linear tape, and `backward()` walks the tape in reverse to propagate gradients via the chain rule.

## Core Data Structures

### Tensor

```rust
pub struct Tensor {
    pub data: Vec<f64>,       // forward values
    pub shape: Vec<usize>,    // dimension sizes
    pub grad: Vec<f64>,       // accumulated gradients
    pub requires_grad: bool,  // true for learnable parameters
    pub op: Op,               // which operation produced this tensor
}
```

### Tape

```rust
pub struct Tape {
    pub tensors: Vec<Tensor>,
    pub param_count: usize,   // boundary between persistent params and ephemeral intermediates
}
```

The tape is a flat `Vec<Tensor>`. Each tensor is identified by its index (`usize`). Operations consume tensor indices and produce new tensor indices.

## The param_count Boundary

The tape splits into two zones:

```
tensors[0 .. param_count]       <- Frozen parameters (weights, biases, embeddings)
                                   Created before tape.freeze_params()
                                   Persist across tape.reset() calls
                                   requires_grad = true

tensors[param_count .. len]     <- Ephemeral intermediates (activations, loss)
                                   Created during forward/backward
                                   Destroyed by tape.reset() each batch
```

This boundary is critical:
- **Optimizers (SGD/Adam) only iterate `[0..param_count]`** to update parameters
- `tape.reset()` truncates the tensor list back to `param_count`, preserving learned weights while clearing all intermediate computation

## Supported Operations

| Op | Forward | Backward |
|----|---------|----------|
| **MatMul** | `C = A @ B` | `dA = dC @ B^T`, `dB = A^T @ dC` |
| **Add** | `C = A + B` (element-wise) | `dA += dC`, `dB += dC` |
| **AddBroadcast** | `C[i,j] = A[i,j] + bias[j]` | `dA += dC`, `dbias[j] += sum_i(dC[i,j])` |
| **ReLU** | `y = max(0, x)` | `dx = (x > 0) ? dy : 0` |
| **MSE** | `L = mean((pred - target)^2)` | `dpred = 2(pred - target) / n` |
| **Conv2D** | 6-nested loop convolution | Input grad + kernel grad via correlation |
| **MaxPool2D** | Max over pool windows | Gradient routed to max indices |
| **Flatten** | Reshape spatial dims to vector | Reshape gradient back |
| **Softmax** | Row-wise exp / sum(exp) | Jacobian-vector product per row |
| **SoftmaxCrossEntropy** | Fused softmax + NLL loss | `dpred = softmax(pred) - target` |
| **LayerNorm** | Per-row normalize + affine | Full Jacobian through mean/var |
| **Embedding** | Gather rows from table | Scatter-add gradients to table rows |
| **Transpose** | Swap dimensions | Transpose gradient back |
| **ScalarMul** | `y = x * c` | `dx = dy * c` |
| **SliceCols** | Extract column range | Scatter gradient to column range |
| **ConcatCols** | Horizontally concatenate tensors | Slice gradient to each input |
| **MeanPool** | Average across rows | Broadcast gradient / num_rows |

## Backward Pass

`tape.backward(loss_idx)` performs:

1. Set `grad[loss_idx] = 1.0` (seed gradient)
2. Walk tensor indices from `loss_idx` down to `0`
3. For each tensor, match on its `Op` variant and propagate gradients to parent tensors
4. Gradients accumulate additively (supporting multi-use tensors)

## Optimizers

### SGD

```
param -= lr * grad
grad = 0
```

Iterates only `tensors[0..param_count]`.

### Adam

Maintains per-parameter first moment (`m`) and second moment (`v`) vectors:

```
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

Bias correction via `t` (timestep counter). State vectors are sized to `param_count`, not the full tape.

## Weight Initialization

All weights use **Kaiming/He initialization**:

```
scale = sqrt(2.0 / fan_in)
weight = xorshift_random() * scale
```

Biases are initialized to zero. The xorshift RNG with seed=42 ensures reproducible initialization across runs.

## Layer Abstractions (`layers.rs`)

Layers are thin wrappers that own parameter indices and call tape operations:

| Layer | Parameters | Forward |
|-------|-----------|---------|
| **Linear** | weights `[in, out]`, bias `[out]` | `matmul + add_broadcast` |
| **Conv2D** | kernel `[out, in, kH, kW]`, bias `[out]` | `tape.conv2d(...)` |
| **Embedding** | table `[vocab, d_model]` | `tape.embedding(...)` |
| **LayerNorm** | gamma `[cols]`, beta `[cols]` | `tape.layernorm(...)` |
| **MultiHeadAttention** | W_Q, W_K, W_V, W_O (all Linear) | Project, split heads, scaled dot-product attention, concat, project |
| **TransformerBlock** | MHA + LayerNorm + FFN + LayerNorm | Attention + residual + norm + FFN + residual + norm |
| **Transformer** | Embedding + PosEnc + N blocks + output Linear | Full encoder stack |
| **VisionTransformer** | Patch projection + PosEnc + N blocks + classifier | Row-wise patches through transformer |
