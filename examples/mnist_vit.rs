//! MNIST classification with a Vision Transformer (ViT).
//!
//! Architecture:
//!   Image[28,28] -> Linear(28->32) -> +PosEnc
//!                -> TransformerBlock x2 -> MeanPool -> Linear(32->10)
//!
//! Uses Adam optimizer and softmax cross-entropy loss.
//!
//! Run: cargo run --release --example mnist_vit

use rust_ann::layers::VisionTransformer;
use rust_ann::mnist::load_mnist;
use rust_ann::tensor::{Adam, Tape};
use std::time::Instant;

fn one_hot(label: u8, num_classes: usize) -> Vec<f64> {
    let mut v = vec![0.0; num_classes];
    v[label as usize] = 1.0;
    v
}

fn argmax(data: &[f64]) -> usize {
    let mut max_idx = 0;
    let mut max_val = data[0];
    for (i, &val) in data.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

fn main() {
    println!("========================================");
    println!("  VISION TRANSFORMER - MNIST");
    println!("========================================");

    let data = load_mnist("data/MNIST/raw");

    let d_model = 32;
    let n_heads = 4;
    let d_ff = 64;
    let n_layers = 2;
    let num_classes = 10;
    let seq_len = 28;
    let patch_size = 28;

    let mut tape = Tape::new();
    let mut seed: u64 = 42;

    let model = VisionTransformer::new(
        &mut tape,
        patch_size,
        seq_len,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        num_classes,
        &mut seed,
    );
    tape.freeze_params();

    let total_params: usize = tape
        .tensors
        .iter()
        .filter(|t| t.requires_grad)
        .map(|t| t.data.len())
        .sum();

    println!("Architecture:");
    println!(
        "  Image [28,28] → Linear(28→{}) → +PE → Transformer×{} → MeanPool → Linear({}→10)",
        d_model, n_layers, d_model
    );
    println!(
        "  d_model={}, n_heads={}, d_ff={}",
        d_model, n_heads, d_ff
    );
    println!("  Total parameters: {}", total_params);

    let epochs = 3;
    let train_samples = 5000;
    let test_samples = 1000;

    let mut optimizer = Adam::new(&tape, 0.001);

    println!("  Optimizer: Adam (lr=0.001, β1=0.9, β2=0.999)");
    println!(
        "  Train samples: {}, Test samples: {}",
        train_samples, test_samples
    );
    println!("---");

    let start_time = Instant::now();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for i in 0..train_samples {
            tape.reset();

            let output = model.forward(&mut tape, &data.train_images[i]);

            let target_data = one_hot(data.train_labels[i], num_classes);
            let target = tape.add_tensor(target_data, vec![1, num_classes], false);

            let loss = tape.softmax_cross_entropy(output, target, num_classes);
            total_loss += tape.tensors[loss].data[0];

            let pred_data = &tape.tensors[output].data;
            let predicted = argmax(&pred_data[0..num_classes]);
            if predicted == data.train_labels[i] as usize {
                correct += 1;
            }

            tape.backward(loss);
            optimizer.step(&mut tape);

            if i % 500 == 0 {
                println!(
                    "  epoch {}/{} sample {}/{} loss: {:.4}",
                    epoch + 1,
                    epochs,
                    i,
                    train_samples,
                    tape.tensors[loss].data[0]
                );
            }
        }

        let avg_loss = total_loss / train_samples as f64;
        let train_accuracy = 100.0 * correct as f64 / train_samples as f64;

        let mut test_correct = 0;
        for i in 0..test_samples {
            tape.reset();
            let output = model.forward(&mut tape, &data.test_images[i]);
            let pred_data = &tape.tensors[output].data;
            let predicted = argmax(&pred_data[0..num_classes]);
            if predicted == data.test_labels[i] as usize {
                test_correct += 1;
            }
        }
        let test_accuracy = 100.0 * test_correct as f64 / test_samples as f64;

        println!(
            "Epoch {}/{}: avg_loss={:.4}, train_acc={:.2}%, test_acc={:.2}%",
            epoch + 1,
            epochs,
            avg_loss,
            train_accuracy,
            test_accuracy
        );
        println!("---");
    }

    let elapsed = start_time.elapsed();
    println!(
        "Vision Transformer training time: {:.1}s",
        elapsed.as_secs_f64()
    );
}
