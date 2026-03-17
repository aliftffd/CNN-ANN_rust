# rust_ann

A from-scratch deep learning framework in pure Rust — **zero external dependencies**. Implements both a fully connected ANN and a CNN with automatic differentiation, trained and compared on MNIST handwritten digits.

## Why

Most ML frameworks are Python + C++. This project proves you can build a working autograd engine, convolutional layers, pooling, and SGD training loop in pure Rust with nothing but `std`. It provides a fair head-to-head comparison between a simple ANN and a CNN on the same dataset under identical conditions.

## Architecture

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

### Autograd Engine (`tensor.rs`)
- Tape-based reverse-mode automatic differentiation
- Supports: MatMul, Conv2D, MaxPool2D, ReLU, MSE, Broadcast Add, Flatten
- SGD optimizer with per-parameter gradient tracking
- Kaiming/He weight initialization (xorshift RNG, seed=42)

## Fair Comparison Results

Both models trained for 3 epochs on the full MNIST dataset (60k train / 10k test) with learning rate 0.01 and data shuffling enabled.

| Metric              |        ANN |        CNN |
|---------------------|------------|------------|
| Parameters          |    101,770 |      5,226 |
| Batch size          |         32 |          1 |
| Data shuffling      |        Yes |        Yes |
| Final train acc     |     67.17% |     96.04% |
| **Final test acc**  | **73.40%** | **96.92%** |
| Training time       |      49.4s |      97.2s |

**CNN achieves ~23% higher test accuracy with ~20x fewer parameters.**

The CNN takes longer to train because it processes one image at a time (no batched convolution), but its spatial feature extraction dramatically outperforms the flat ANN.

## Project Structure

```
rust_ann/
├── Cargo.toml          # Zero dependencies
├── src/
│   ├── main.rs         # Training loops, comparison, shuffling
│   ├── tensor.rs       # Autograd tape, forward/backward ops, SGD
│   ├── layers.rs       # Linear and Conv2D layer abstractions
│   └── mnist.rs        # MNIST binary format loader
└── data/
    └── MNIST/raw/      # Dataset files (not in git, see Setup)
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
cargo run --release
```

Release mode is recommended — debug builds are significantly slower for the matrix operations.

## How It Works

### Training Loop
1. Load and normalize MNIST images to `[0.0, 1.0]`
2. Shuffle training indices each epoch (Fisher-Yates with xorshift RNG)
3. Forward pass through network layers
4. Compute MSE loss against one-hot targets
5. Backward pass through the computation graph (reverse topological order)
6. SGD parameter update: `param -= lr * grad`
7. Evaluate on full test set after each epoch

### Autograd
Each operation records itself on a `Tape`. During `backward()`, gradients flow in reverse through the graph using the chain rule. The tape is reset between batches/samples while preserving learned parameters via `freeze_params()`.

### Key Design Decisions
- **MSE loss** instead of cross-entropy — simpler gradient computation, still converges well
- **No softmax** — raw logits compared via argmax for classification
- **Deterministic RNG** (seed=42) — reproducible weight initialization across runs
- **Tape-based autograd** — clean separation between forward computation and gradient tracking

## Requirements

- Rust 1.56+ (2021 edition)
- ~64MB disk for MNIST data
- No external crates
