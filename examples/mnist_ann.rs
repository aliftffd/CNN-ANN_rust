//! MNIST classification with a fully-connected ANN.
//!
//! Architecture: 784 -> 128 (ReLU) -> 10
//! Loss: MSE, Optimizer: SGD, Batch size: 32
//!
//! Run: cargo run --release --example mnist_ann

use rust_ann::layers::Linear;
use rust_ann::mnist::load_mnist;
use rust_ann::tensor::{sgd_step, Tape};
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

/// Fisher-Yates shuffle using xorshift RNG (no external deps)
fn shuffle_indices(indices: &mut [usize], seed: &mut u64) {
    let n = indices.len();
    for i in (1..n).rev() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (*seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

fn main() {
    println!("========================================");
    println!("  TRAINING ANN (784 -> 128 -> 10)");
    println!("========================================");

    let data = load_mnist("data/MNIST/raw");

    let mut tape = Tape::new();
    let mut seed: u64 = 42;
    let layer1 = Linear::new(&mut tape, 784, 128, &mut seed);
    let layer2 = Linear::new(&mut tape, 128, 10, &mut seed);
    tape.freeze_params();

    let lr = 0.01;
    let batch_size = 32;
    let epochs = 3;
    let num_train = data.train_images.len();
    let num_batches = num_train / batch_size;

    let total_params = 784 * 128 + 128 + 128 * 10 + 10;
    println!("Parameters: {}", total_params);
    println!(
        "Learning rate: {}, Batch size: {}, Epochs: {}",
        lr, batch_size, epochs
    );
    println!(
        "Train samples: {}, Test samples: {}",
        num_train,
        data.test_images.len()
    );
    println!("Data shuffling: YES");
    println!("---");

    let start_time = Instant::now();

    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut shuffle_seed: u64 = 12345;

    for epoch in 0..epochs {
        shuffle_indices(&mut indices, &mut shuffle_seed);

        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for batch in 0..num_batches {
            tape.reset();

            let mut batch_input = Vec::with_capacity(batch_size * 784);
            let mut batch_target = Vec::with_capacity(batch_size * 10);

            for b in 0..batch_size {
                let idx = indices[batch * batch_size + b];
                batch_input.extend_from_slice(&data.train_images[idx]);
                batch_target.extend(one_hot(data.train_labels[idx], 10));
            }

            let x = tape.add_tensor(batch_input, vec![batch_size, 784], false);
            let target = tape.add_tensor(batch_target, vec![batch_size, 10], false);

            let z1 = layer1.forward(&mut tape, x);
            let a1 = tape.relu(z1);
            let pred = layer2.forward(&mut tape, a1);

            let loss = tape.mse_loss(pred, target);
            total_loss += tape.tensors[loss].data[0];

            for b in 0..batch_size {
                let pred_slice = &tape.tensors[pred].data[b * 10..(b + 1) * 10];
                let predicted_label = argmax(pred_slice);
                let idx = indices[batch * batch_size + b];
                let actual_label = data.train_labels[idx] as usize;
                if predicted_label == actual_label {
                    correct += 1;
                }
                total += 1;
            }

            tape.backward(loss);
            sgd_step(&mut tape, lr);

            if batch % 500 == 0 {
                println!(
                    "  epoch {}/{} batch {}/{} loss: {:.4}",
                    epoch + 1,
                    epochs,
                    batch,
                    num_batches,
                    tape.tensors[loss].data[0]
                );
            }
        }

        let avg_loss = total_loss / num_batches as f64;
        let accuracy = 100.0 * correct as f64 / total as f64;

        // Test accuracy
        let mut test_correct = 0;
        let test_batches = data.test_images.len() / batch_size;
        for batch in 0..test_batches {
            tape.reset();
            let start = batch * batch_size;
            let mut batch_input = Vec::with_capacity(batch_size * 784);
            for i in start..start + batch_size {
                batch_input.extend_from_slice(&data.test_images[i]);
            }
            let x = tape.add_tensor(batch_input, vec![batch_size, 784], false);
            let z1 = layer1.forward(&mut tape, x);
            let a1 = tape.relu(z1);
            let pred = layer2.forward(&mut tape, a1);

            for i in 0..batch_size {
                let pred_slice = &tape.tensors[pred].data[i * 10..(i + 1) * 10];
                let predicted_label = argmax(pred_slice);
                let actual_label = data.test_labels[start + i] as usize;
                if predicted_label == actual_label {
                    test_correct += 1;
                }
            }
        }
        let test_accuracy = 100.0 * test_correct as f64 / (test_batches * batch_size) as f64;

        println!(
            "Epoch {}/{}: avg_loss={:.4}, train_acc={:.2}%, test_acc={:.2}%",
            epoch + 1,
            epochs,
            avg_loss,
            accuracy,
            test_accuracy
        );
        println!("---");
    }

    let elapsed = start_time.elapsed();
    println!("ANN total time: {:.1}s", elapsed.as_secs_f64());
}
