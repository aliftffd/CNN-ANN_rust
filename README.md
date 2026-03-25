# rust_ann

A from-scratch deep learning framework in pure Rust — **zero external dependencies**. Implements an ANN, CNN, Transformer, and Vision Transformer with automatic differentiation, all built on a single tape-based autograd engine.

## Why

Most ML frameworks are Python + C++. This project proves you can build a working autograd engine, convolutional layers, multi-head attention, layer normalization, and full training loops in pure Rust with nothing but `std`. It provides a head-to-head comparison across four architectures on the same training infrastructure.

## Architectures

### ANN (Artificial Neural Network)
```
Input(784) -> Linear(128) -> ReLU -> Linear(10) -> MSE Loss
```
- **101,770 parameters**
- Batched training (batch size 32)
- Flattened 28x28 images to 784-dim vectors

### CNN (Convolutional Neural Network)
```
Input(1x28x28) -> Conv2D(1->8, 3x3) -> ReLU -> MaxPool(2x2)
              -> Conv2D(8->16, 3x3) -> ReLU -> MaxPool(2x2)
              -> Flatten(400) -> Linear(10) -> MSE Loss
```
- **5,226 parameters** (~20x fewer than ANN)
- Single-image processing (no batching)
- Preserves spatial structure of images

### Transformer (Character Prediction)
```
Embedding(6, 32) -> +PosEnc -> TransformerBlock x2 -> Linear(32->6) -> Softmax + CrossEntropy
```
Each TransformerBlock:
```
MultiHeadAttention(4 heads) -> Residual + LayerNorm -> FFN(32->64->32) -> Residual + LayerNorm
```
- **d_model=32, n_heads=4, d_ff=64, 2 layers**
- Trained on next-character prediction over a repeating `abcdef` pattern
- Autoregressive generation after training

### Vision Transformer (ViT)
```
Image[28,28] -> Linear(28->32) -> +PosEnc -> TransformerBlock x2 -> MeanPool -> Linear(32->10)
```
- Row-wise patch embedding (each 28-pixel row is a patch)
- Trained on MNIST with Adam optimizer and softmax cross-entropy loss
- Separate binary: `cargo run --release --bin main_vit`

## Autograd Engine (`tensor.rs`)

Tape-based reverse-mode automatic differentiation supporting:

| Category | Operations |
|----------|-----------|
| Linear algebra | MatMul, Add, Mul, Transpose, ScalarMul |
| Convolution | Conv2D, MaxPool2D, Flatten |
| Activations | ReLU, Softmax |
| Losses | MSE, Softmax + CrossEntropy |
| Normalization | LayerNorm |
| Attention | SliceCols, ConcatCols, MeanPool |
| Sequence | Embedding, Positional Encoding |
| Optimizers | SGD, Adam (with bias-corrected first/second moments) |

Weight initialization uses Kaiming/He scaling with a deterministic xorshift RNG (seed=42).

## MNIST Comparison Results

ANN and CNN trained for 3 epochs on full MNIST (60k train / 10k test), learning rate 0.01, data shuffling enabled.

| Metric              |        ANN |        CNN |
|---------------------|------------|------------|
| Parameters          |    101,770 |      5,226 |
| Batch size          |         32 |          1 |
| Data shuffling      |        Yes |        Yes |
| Final train acc     |     67.17% |     96.04% |
| **Final test acc**  | **73.40%** | **96.92%** |
| Training time       |      49.4s |      97.2s |

**CNN achieves ~23% higher test accuracy with ~20x fewer parameters.**

## Project Structure

```
rust_ann/
├── Cargo.toml              # Zero dependencies, two binaries
├── src/
│   ├── lib.rs              # Crate root (re-exports modules)
│   ├── tensor.rs           # Autograd tape, forward/backward ops, SGD, Adam
│   ├── layers.rs           # Linear, Conv2D, Embedding, MultiHeadAttention,
│   │                       #   LayerNorm, TransformerBlock, Transformer, VisionTransformer
│   ├── mnist.rs            # MNIST binary format loader
│   ├── main.rs             # ANN + CNN + Transformer training
│   ├── main_vit.rs         # Vision Transformer training (separate binary)
│   └── transformer_train.rs # Transformer character-prediction training loop
└── data/
    └── MNIST/raw/          # Dataset files (not in git, see Setup)
```

## Setup

### 1. Clone
```bash
git clone <repo-url>
cd rust_ann
```

### 2. Download MNIST
```bash
mkdir -p data/MNIST/raw && cd data/MNIST/raw

# Download the 4 dataset files
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    curl -O "https://storage.googleapis.com/cvdf-datasets/mnist/${f}.gz"
    gunzip "${f}.gz"
done
```

### 3. Build and Run
```bash
# ANN + CNN + Transformer comparison
cargo run --release

# Vision Transformer (MNIST)
cargo run --release --bin main_vit
```

Release mode is recommended — debug builds are significantly slower for the matrix operations.

## How It Works

### Training Loop
1. Load and normalize MNIST images to `[0.0, 1.0]`
2. Shuffle training indices each epoch (Fisher-Yates with xorshift RNG)
3. Forward pass through network layers
4. Compute loss (MSE for ANN/CNN, softmax cross-entropy for Transformer/ViT)
5. Backward pass through the computation graph (reverse topological order)
6. Parameter update via SGD or Adam
7. Evaluate on test set after each epoch

### Autograd
Each operation records itself on a `Tape`. During `backward()`, gradients flow in reverse through the graph using the chain rule. The tape is reset between batches/samples while preserving learned parameters via `freeze_params()`.

### Key Design Decisions
- **MSE loss** for ANN/CNN, **softmax cross-entropy** for Transformer/ViT
- **Deterministic RNG** (seed=42) — reproducible weight initialization across runs
- **Tape-based autograd** — clean separation between forward computation and gradient tracking
- **Adam optimizer** with bias-corrected moments for the ViT pipeline
- **Row-wise patch embedding** in ViT — each image row is treated as a token

## Requirements

- Rust 1.56+ (2021 edition)
- ~64MB disk for MNIST data
- No external crates
