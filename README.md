# rust_ann

A from-scratch deep learning framework in pure Rust — **zero external dependencies**. Implements an ANN, CNN, Transformer, and Vision Transformer with automatic differentiation, all built on a single tape-based autograd engine.

## Why

Most ML frameworks are Python + C++. This project proves you can build a working autograd engine, convolutional layers, multi-head attention, layer normalization, and full training loops in pure Rust with nothing but `std`.

## Quick Start

```bash
# Download MNIST (see Setup below)

# Run examples
cargo run --release --example mnist_ann
cargo run --release --example mnist_cnn
cargo run --release --example mnist_vit
cargo run --release --example char_transformer
cargo run --release --example xor
```

## Architectures

### ANN — Fully Connected Network
```
Input(784) -> Linear(128) -> ReLU -> Linear(10) -> MSE Loss
```
101,770 parameters | Batched (32) | **~73% test accuracy**

### CNN — Convolutional Network
```
Input(1x28x28) -> Conv2D(1->8,3x3) -> ReLU -> MaxPool(2x2)
              -> Conv2D(8->16,3x3) -> ReLU -> MaxPool(2x2)
              -> Flatten(400) -> Linear(10) -> MSE Loss
```
5,226 parameters | Single-image | **~97% test accuracy**

### Transformer — Character Prediction
```
Embedding(6,32) -> +PosEnc -> TransformerBlock x2 -> Linear(32->6) -> Softmax+CE
```
Learns to predict next character in a repeating `abcdef` pattern with autoregressive generation.

### Vision Transformer (ViT) — MNIST
```
Image[28,28] -> Linear(28->32) -> +PosEnc -> TransformerBlock x2 -> MeanPool -> Linear(32->10)
```
Row-wise patch embedding | Adam optimizer | Softmax cross-entropy loss

## Project Structure

```
rust_ann/
├── README.md
├── Cargo.toml
├── .cargo/config.toml          # target-cpu=native for SIMD
├── docs/
│   ├── architecture.md         # How the autograd engine works
│   ├── benchmarks.md           # ANN vs CNN vs ViT comparison
│   └── cuda_roadmap.md         # CUDA acceleration plan
├── src/
│   ├── lib.rs                  # Crate root
│   ├── tensor.rs               # Autograd tape, forward/backward ops, SGD, Adam
│   ├── layers.rs               # Linear, Conv2D, Embedding, MHA, LayerNorm,
│   │                           #   TransformerBlock, Transformer, VisionTransformer
│   └── mnist.rs                # MNIST binary format loader
├── examples/
│   ├── xor.rs                  # Simplest demo — learn XOR gate
│   ├── mnist_ann.rs            # Fully connected ANN on MNIST
│   ├── mnist_cnn.rs            # CNN on MNIST
│   ├── mnist_vit.rs            # Vision Transformer on MNIST
│   └── char_transformer.rs     # Transformer character prediction
└── assets/
    └── (training curves, diagrams)
```

## Autograd Engine

Tape-based reverse-mode automatic differentiation. See [docs/architecture.md](docs/architecture.md) for full details.

| Category | Operations |
|----------|-----------|
| Linear algebra | MatMul, Add, Mul, Transpose, ScalarMul |
| Convolution | Conv2D, MaxPool2D, Flatten |
| Activations | ReLU, Softmax |
| Losses | MSE, Softmax + CrossEntropy |
| Normalization | LayerNorm |
| Attention | SliceCols, ConcatCols, MeanPool |
| Sequence | Embedding, Positional Encoding |
| Optimizers | SGD, Adam |

## Setup

### 1. Clone
```bash
git clone <repo-url>
cd rust_ann
```

### 2. Download MNIST
```bash
mkdir -p data/MNIST/raw && cd data/MNIST/raw

for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -O "https://storage.googleapis.com/cvdf-datasets/mnist/${f}.gz"
    gunzip "${f}.gz"
done
```

### 3. Build and Run
```bash
cargo run --release --example mnist_ann
```

Release mode is recommended — debug builds are significantly slower for the matrix operations.

## Documentation

- **[Architecture](docs/architecture.md)** — How the autograd tape, backward pass, and optimizers work
- **[Benchmarks](docs/benchmarks.md)** — ANN vs CNN vs Transformer vs ViT comparison results
- **[CUDA Roadmap](docs/cuda_roadmap.md)** — Plan for GPU acceleration via cudarc/cuBLAS/cuDNN

## Requirements

- Rust 1.56+ (2021 edition)
- ~64MB disk for MNIST data
- No external crates
