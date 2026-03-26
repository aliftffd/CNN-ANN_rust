//! Character-level Transformer for next-character prediction.
//!
//! Task: Learn the repeating pattern "abcdefabcdef..."
//! Architecture:
//!   Embedding(6, 32) -> +PosEnc -> TransformerBlock x2 -> Linear(32->6)
//!
//! Uses SGD optimizer and softmax cross-entropy loss.
//!
//! Run: cargo run --release --example char_transformer

use rust_ann::layers::Transformer;
use rust_ann::tensor::{sgd_step, Tape};
use std::time::Instant;

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

fn one_hot_sequence(targets: &[usize], vocab_size: usize) -> Vec<f64> {
    let mut data = vec![0.0; targets.len() * vocab_size];
    for (i, &t) in targets.iter().enumerate() {
        data[i * vocab_size + t] = 1.0;
    }
    data
}

fn main() {
    println!("========================================");
    println!("  TRANSFORMER - Character Prediction");
    println!("========================================");

    // Task: learn to predict next character in a repeating pattern
    // Pattern: "abcdefabcdef..." (6 unique characters)
    let vocab_size = 6; // a=0, b=1, c=2, d=3, e=4, f=5
    let pattern: Vec<usize> = vec![0, 1, 2, 3, 4, 5];

    // Generate training data: repeat the pattern
    let num_repeats = 20;
    let full_sequence: Vec<usize> = pattern
        .iter()
        .cycle()
        .take(pattern.len() * num_repeats)
        .cloned()
        .collect();

    // Training sequences: sliding window
    let seq_len = 8;
    let mut train_inputs: Vec<Vec<usize>> = Vec::new();
    let mut train_targets: Vec<Vec<usize>> = Vec::new();

    for i in 0..full_sequence.len() - seq_len {
        let input = full_sequence[i..i + seq_len].to_vec();
        let target = full_sequence[i + 1..i + seq_len + 1].to_vec();
        train_inputs.push(input);
        train_targets.push(target);
    }

    println!("Training samples: {}", train_inputs.len());
    println!("Sequence length: {}", seq_len);
    println!(
        "Vocab size: {} (a=0, b=1, c=2, d=3, e=4, f=5)",
        vocab_size
    );

    // Create Transformer
    let d_model = 32;
    let n_heads = 4;
    let d_ff = 64;
    let n_layers = 2;
    let max_seq_len = 64;

    let mut tape = Tape::new();
    let mut seed: u64 = 42;

    let transformer = Transformer::new(
        &mut tape,
        vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        max_seq_len,
        &mut seed,
    );
    tape.freeze_params();

    println!(
        "Model: d_model={}, n_heads={}, d_ff={}, n_layers={}",
        d_model, n_heads, d_ff, n_layers
    );
    println!("---");

    // Training loop
    let lr = 0.001;
    let epochs = 50;
    let num_samples = train_inputs.len();

    println!("Learning rate: {}", lr);
    println!("Epochs: {}", epochs);
    println!("---");

    let start_time = Instant::now();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for s in 0..num_samples {
            tape.reset();

            let output = transformer.forward(&mut tape, &train_inputs[s]);

            let target_data = one_hot_sequence(&train_targets[s], vocab_size);
            let target = tape.add_tensor(target_data, vec![seq_len, vocab_size], false);

            let loss = tape.softmax_cross_entropy(output, target, vocab_size);
            total_loss += tape.tensors[loss].data[0];

            let pred_data = &tape.tensors[output].data;
            for pos in 0..seq_len {
                let start = pos * vocab_size;
                let predicted = argmax(&pred_data[start..start + vocab_size]);
                if predicted == train_targets[s][pos] {
                    correct += 1;
                }
                total += 1;
            }

            tape.backward(loss);
            sgd_step(&mut tape, lr);
        }

        let avg_loss = total_loss / num_samples as f64;
        let accuracy = 100.0 * correct as f64 / total as f64;

        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: loss={:.4}, accuracy={:.2}%",
                epoch + 1,
                epochs,
                avg_loss,
                accuracy
            );
        }
    }

    let elapsed = start_time.elapsed();
    println!("Transformer training time: {:.1}s", elapsed.as_secs_f64());

    // Generation test
    println!();
    println!("========================================");
    println!("  GENERATION TEST");
    println!("========================================");

    let mut generated: Vec<usize> = vec![0, 1, 2]; // "abc"
    let chars = ['a', 'b', 'c', 'd', 'e', 'f'];

    print!("Seed: ");
    for &t in &generated {
        print!("{}", chars[t]);
    }
    println!();

    // Generate 20 more characters autoregressively
    for _ in 0..20 {
        tape.reset();

        let context_start = if generated.len() > seq_len {
            generated.len() - seq_len
        } else {
            0
        };
        let context = &generated[context_start..];

        let output = transformer.forward(&mut tape, context);

        let last_pos = context.len() - 1;
        let start = last_pos * vocab_size;
        let pred_data = &tape.tensors[output].data;
        let next_token = argmax(&pred_data[start..start + vocab_size]);

        generated.push(next_token);
    }

    print!("Generated: ");
    for &t in &generated {
        print!("{}", chars[t]);
    }
    println!();

    print!("Expected:  ");
    for i in 0..generated.len() {
        print!("{}", chars[i % pattern.len()]);
    }
    println!();

    let mut gen_correct = 0;
    for (i, &t) in generated.iter().enumerate() {
        if t == i % pattern.len() {
            gen_correct += 1;
        }
    }
    println!(
        "Generation accuracy: {}/{} ({:.1}%)",
        gen_correct,
        generated.len(),
        100.0 * gen_correct as f64 / generated.len() as f64
    );
}
