//! MNIST classification with a CNN (no batching, 1 image at a time).
//!
//! Architecture:
//!   Conv(1->8, 3x3) -> ReLU -> MaxPool(2)
//!   Conv(8->16, 3x3) -> ReLU -> MaxPool(2)
//!   Flatten -> Linear(400->10)
//!
//! Loss: MSE, Optimizer: SGD
//!
//! Run: cargo run --release --example mnist_cnn

use rust_ann::layers::{Conv2D, Linear};
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
    println!("  TRAINING CNN (no batching, 1 image at a time)");
    println!("  Conv(1->8,3x3) -> Pool -> Conv(8->16,3x3) -> Pool -> Linear(400->10)");
    println!("========================================");

    let data = load_mnist("data/MNIST/raw");

    let mut tape = Tape::new();
    let mut seed: u64 = 42;

    let conv1 = Conv2D::new(&mut tape, 1, 8, 3, 3, &mut seed);
    let conv2 = Conv2D::new(&mut tape, 8, 16, 3, 3, &mut seed);
    let fc = Linear::new(&mut tape, 400, 10, &mut seed);
    tape.freeze_params();

    let total_params = (1 * 8 * 3 * 3 + 8) + (8 * 16 * 3 * 3 + 16) + (400 * 10 + 10);
    println!("Parameters: {}", total_params);

    let lr = 0.01;
    let epochs = 3;
    let num_train = data.train_images.len();
    let num_test = data.test_images.len();

    println!(
        "Learning rate: {}, Batch size: 1 (no batching), Epochs: {}",
        lr, epochs
    );
    println!("Train samples: {}, Test samples: {}", num_train, num_test);
    println!("Data shuffling: YES");
    println!("---");

    let start_time = Instant::now();

    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut shuffle_seed: u64 = 12345;

    for epoch in 0..epochs {
        shuffle_indices(&mut indices, &mut shuffle_seed);

        let mut total_loss = 0.0;
        let mut correct = 0;

        for (step, &idx) in indices.iter().enumerate() {
            tape.reset();

            let x = tape.add_tensor(data.train_images[idx].clone(), vec![1, 28, 28], false);
            let target =
                tape.add_tensor(one_hot(data.train_labels[idx], 10), vec![1, 10], false);

            let c1 = conv1.forward(&mut tape, x, 28, 28);
            let a1 = tape.relu(c1);
            let p1 = tape.max_pool2d(a1, 8, 26, 26, 2);

            let c2 = conv2.forward(&mut tape, p1, 13, 13);
            let a2 = tape.relu(c2);
            let p2 = tape.max_pool2d(a2, 16, 11, 11, 2);

            let flat = tape.flatten_2d(p2, 16, 5, 5);
            let pred = fc.forward(&mut tape, flat);

            let loss = tape.mse_loss(pred, target);
            total_loss += tape.tensors[loss].data[0];

            let pred_slice = &tape.tensors[pred].data[0..10];
            let predicted_label = argmax(pred_slice);
            if predicted_label == data.train_labels[idx] as usize {
                correct += 1;
            }

            tape.backward(loss);
            sgd_step(&mut tape, lr);

            if step % 5000 == 0 {
                println!(
                    "  epoch {}/{} sample {}/{} loss: {:.4}",
                    epoch + 1,
                    epochs,
                    step,
                    num_train,
                    tape.tensors[loss].data[0]
                );
            }
        }

        let avg_loss = total_loss / num_train as f64;
        let train_accuracy = 100.0 * correct as f64 / num_train as f64;

        // Test accuracy
        let mut test_correct = 0;
        for i in 0..num_test {
            tape.reset();
            let x = tape.add_tensor(data.test_images[i].clone(), vec![1, 28, 28], false);

            let c1 = conv1.forward(&mut tape, x, 28, 28);
            let a1 = tape.relu(c1);
            let p1 = tape.max_pool2d(a1, 8, 26, 26, 2);

            let c2 = conv2.forward(&mut tape, p1, 13, 13);
            let a2 = tape.relu(c2);
            let p2 = tape.max_pool2d(a2, 16, 11, 11, 2);

            let flat = tape.flatten_2d(p2, 16, 5, 5);
            let pred = fc.forward(&mut tape, flat);

            let pred_slice = &tape.tensors[pred].data[0..10];
            let predicted_label = argmax(pred_slice);
            if predicted_label == data.test_labels[i] as usize {
                test_correct += 1;
            }
        }
        let test_accuracy = 100.0 * test_correct as f64 / num_test as f64;

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
    println!("CNN total time: {:.1}s", elapsed.as_secs_f64());
}
