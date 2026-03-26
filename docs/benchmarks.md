# Benchmarks

## MNIST: ANN vs CNN

Both models trained for 3 epochs on the full MNIST dataset (60k train / 10k test) with learning rate 0.01 and data shuffling enabled.

| Metric              |        ANN |        CNN |
|---------------------|------------|------------|
| Parameters          |    101,770 |      5,226 |
| Batch size          |         32 |          1 |
| Data shuffling      |        Yes |        Yes |
| Loss function       |        MSE |        MSE |
| Optimizer           |        SGD |        SGD |
| Final train acc     |     67.17% |     96.04% |
| **Final test acc**  | **73.40%** | **96.92%** |
| Training time       |      49.4s |      97.2s |

**CNN achieves ~23% higher test accuracy with ~20x fewer parameters.**

The CNN takes longer to train because it processes one image at a time (no batched convolution), but its spatial feature extraction dramatically outperforms the flat ANN.

### ANN Architecture
```
Input(784) -> Linear(128) -> ReLU -> Linear(10) -> MSE Loss
```

### CNN Architecture
```
Input(1x28x28) -> Conv2D(1->8, 3x3) -> ReLU -> MaxPool(2x2)
              -> Conv2D(8->16, 3x3) -> ReLU -> MaxPool(2x2)
              -> Flatten(400) -> Linear(10) -> MSE Loss
```

## Transformer: Character Prediction

Trained on next-character prediction over a repeating `abcdef` pattern with sliding window.

| Setting | Value |
|---------|-------|
| Vocab size | 6 (a-f) |
| Sequence length | 8 |
| d_model | 32 |
| Heads | 4 |
| FFN dim | 64 |
| Layers | 2 |
| Optimizer | SGD (lr=0.001) |
| Epochs | 50 |

After training, the model generates the correct `abcdef` cycle autoregressively from a seed of `abc`.

## Vision Transformer (ViT): MNIST

| Setting | Value |
|---------|-------|
| Patch embedding | Row-wise (28 rows x 28 pixels) |
| d_model | 32 |
| Heads | 4 |
| FFN dim | 64 |
| Layers | 2 |
| Optimizer | Adam (lr=0.001) |
| Loss | Softmax + CrossEntropy |
| Train subset | 5,000 samples |
| Test subset | 1,000 samples |

## Environment

All benchmarks measured on CPU, compiled with `--release`. Timings will vary by hardware. The `.cargo/config.toml` enables `-C target-cpu=native` for SIMD autovectorization.
