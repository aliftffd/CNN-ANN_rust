use rust_ann::layers::{Conv2D, Linear};
use rust_ann::mnist::{self, load_mnist};
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
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (*seed >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
}

struct TrainResult {
    final_train_acc: f64,
    final_test_acc: f64,
    elapsed_secs: f64,
}

fn train_ann(data: &mnist::MnistData) -> TrainResult {
    println!("========================================");
    println!("  TRAINING ANN (784 -> 128 -> 10)");
    println!("========================================");

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
    println!("Learning rate: {}, Batch size: {}, Epochs: {}", lr, batch_size, epochs);
    println!("Train samples: {}, Test samples: {}", num_train, data.test_images.len());
    println!("Data shuffling: YES");
    println!("---");

    let start_time = Instant::now();

    // Shuffleable index array
    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut shuffle_seed: u64 = 12345;

    let mut final_train_acc = 0.0;
    let mut final_test_acc = 0.0;

    for epoch in 0..epochs {
        // Shuffle training data each epoch
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

        final_train_acc = accuracy;
        final_test_acc = test_accuracy;

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
    println!();

    TrainResult {
        final_train_acc,
        final_test_acc,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

fn train_cnn(data: &mnist::MnistData) -> TrainResult {
    println!("========================================");
    println!("  TRAINING CNN (no batching, 1 image at a time)");
    println!("  Conv(1->8,3x3) -> Pool -> Conv(8->16,3x3) -> Pool -> Linear(400->10)");
    println!("========================================");

    let mut tape = Tape::new();
    let mut seed: u64 = 42;

    // Conv layers
    let conv1 = Conv2D::new(&mut tape, 1, 8, 3, 3, &mut seed);
    let conv2 = Conv2D::new(&mut tape, 8, 16, 3, 3, &mut seed);
    // Fully connected
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

    // Shuffleable index array
    let mut indices: Vec<usize> = (0..num_train).collect();
    let mut shuffle_seed: u64 = 12345;

    let mut final_train_acc = 0.0;
    let mut final_test_acc = 0.0;

    for epoch in 0..epochs {
        // Shuffle training data each epoch
        shuffle_indices(&mut indices, &mut shuffle_seed);

        let mut total_loss = 0.0;
        let mut correct = 0;

        for (step, &idx) in indices.iter().enumerate() {
            tape.reset();

            // Input: single image [1, 28, 28]
            let x = tape.add_tensor(data.train_images[idx].clone(), vec![1, 28, 28], false);
            let target = tape.add_tensor(one_hot(data.train_labels[idx], 10), vec![1, 10], false);

            // Forward: conv1 -> relu -> pool -> conv2 -> relu -> pool -> flatten -> fc
            let c1 = conv1.forward(&mut tape, x, 28, 28); // [8, 26, 26]
            let a1 = tape.relu(c1); // [8, 26, 26]
            let p1 = tape.max_pool2d(a1, 8, 26, 26, 2); // [8, 13, 13]

            let c2 = conv2.forward(&mut tape, p1, 13, 13); // [16, 11, 11]
            let a2 = tape.relu(c2); // [16, 11, 11]
            let p2 = tape.max_pool2d(a2, 16, 11, 11, 2); // [16, 5, 5]

            let flat = tape.flatten_2d(p2, 16, 5, 5); // [1, 400]
            let pred = fc.forward(&mut tape, flat); // [1, 10]

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

        // Test accuracy (full test set)
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

        final_train_acc = train_accuracy;
        final_test_acc = test_accuracy;

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
    println!();

    TrainResult {
        final_train_acc,
        final_test_acc,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

fn main() {
    let data = load_mnist("data/MNIST/raw");

    println!();
    let ann_result = train_ann(&data);

    println!();
    let cnn_result = train_cnn(&data);

    println!();
    rust_ann::transformer_train::train_transformer();

    println!("========================================");
    println!("  FAIR COMPARISON SUMMARY");
    println!("========================================");
    println!();
    println!("{:<25} {:>12} {:>12}", "", "ANN", "CNN");
    println!("{:-<25} {:-<12} {:-<12}", "", "", "");
    println!(
        "{:<25} {:>12} {:>12}",
        "Architecture",
        "784->128->10",
        "Conv+Pool+FC"
    );
    println!(
        "{:<25} {:>12} {:>12}",
        "Parameters",
        "101,770",
        "5,226"
    );
    println!(
        "{:<25} {:>12} {:>12}",
        "Batch size",
        "32",
        "1"
    );
    println!(
        "{:<25} {:>12} {:>12}",
        "Data shuffling",
        "Yes",
        "Yes"
    );
    println!(
        "{:<25} {:>11.2}% {:>11.2}%",
        "Final train accuracy",
        ann_result.final_train_acc,
        cnn_result.final_train_acc
    );
    println!(
        "{:<25} {:>11.2}% {:>11.2}%",
        "Final test accuracy",
        ann_result.final_test_acc,
        cnn_result.final_test_acc
    );
    println!(
        "{:<25} {:>11.1}s {:>11.1}s",
        "Training time",
        ann_result.elapsed_secs,
        cnn_result.elapsed_secs
    );
    println!();

    let acc_diff = cnn_result.final_test_acc - ann_result.final_test_acc;
    if acc_diff > 0.0 {
        println!(
            "CNN achieves {:.2}% higher test accuracy with ~{:.0}x fewer parameters",
            acc_diff,
            101770.0 / 5226.0
        );
    } else if acc_diff < 0.0 {
        println!(
            "ANN achieves {:.2}% higher test accuracy but uses ~{:.0}x more parameters",
            -acc_diff,
            101770.0 / 5226.0
        );
    } else {
        println!("Both models achieve the same test accuracy!");
    }
    println!(
        "CNN processes 1 image at a time (no batching) vs ANN's batch size of 32"
    );
    println!("========================================");
}
