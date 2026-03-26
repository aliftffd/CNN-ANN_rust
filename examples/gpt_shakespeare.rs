//! GPT decoder trained on a Shakespeare-like corpus.
//!
//! Learns character-level language modeling with causal masking on real English text.
//! After training, generates text from multiple prompts autoregressively.
//!
//! Architecture:
//!   Embedding(vocab, 32) -> +PosEnc -> DecoderBlock x2 (causal mask) -> Linear(32->vocab)
//!
//! Uses Adam optimizer and softmax cross-entropy loss.
//!
//! Run: cargo run --release --example gpt_shakespeare

use rust_ann::layers::GPT;
use rust_ann::tensor::{Adam, Tape};
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
    println!("  GPT DECODER - Shakespeare");
    println!("  (causal mask — autoregressive)");
    println!("========================================");

    // Shakespeare-like corpus
    let corpus = "to be or not to be that is the question \
                  whether tis nobler in the mind to suffer \
                  the slings and arrows of outrageous fortune \
                  or to take arms against a sea of troubles ";

    // Build character vocabulary
    let mut chars: Vec<char> = corpus
        .chars()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    chars.sort();
    let vocab_size = chars.len();

    let char_to_idx: std::collections::HashMap<char, usize> =
        chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let idx_to_char: std::collections::HashMap<usize, char> =
        chars.iter().enumerate().map(|(i, &c)| (i, c)).collect();

    println!("Corpus: {} chars, {} unique", corpus.len(), vocab_size);

    // Tokenize
    let tokens: Vec<usize> = corpus.chars().map(|c| char_to_idx[&c]).collect();

    // Sliding window
    let seq_len = 16;
    let mut train_inputs: Vec<Vec<usize>> = Vec::new();
    let mut train_targets: Vec<Vec<usize>> = Vec::new();

    for i in 0..tokens.len() - seq_len {
        train_inputs.push(tokens[i..i + seq_len].to_vec());
        train_targets.push(tokens[i + 1..i + seq_len + 1].to_vec());
    }

    println!("Training samples: {}", train_inputs.len());
    println!("Sequence length: {}", seq_len);
    println!("Vocab size: {}", vocab_size);

    // Create GPT decoder
    let d_model = 32;
    let n_heads = 4;
    let d_ff = 64;
    let n_layers = 2;
    let max_seq_len = 64;

    let mut tape = Tape::new();
    let mut seed: u64 = 42;

    let gpt = GPT::new(
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

    let total_params: usize = tape
        .tensors
        .iter()
        .filter(|t| t.requires_grad)
        .map(|t| t.data.len())
        .sum();

    println!(
        "Model: GPT d_model={}, n_heads={}, d_ff={}, n_layers={}",
        d_model, n_heads, d_ff, n_layers
    );
    println!("Parameters: {}", total_params);
    println!("Masking: CAUSAL (lower-triangular)");
    println!("Optimizer: Adam (lr=0.001)");
    println!("---");

    // Training loop
    let epochs = 100;
    let num_samples = train_inputs.len();
    let mut optimizer = Adam::new(&tape, 0.001);

    let start_time = Instant::now();

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        for s in 0..num_samples {
            tape.reset();

            let output = gpt.forward(&mut tape, &train_inputs[s]);

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
            optimizer.step(&mut tape);
        }

        let avg_loss = total_loss / num_samples as f64;
        let accuracy = 100.0 * correct as f64 / total as f64;

        if epoch % 10 == 0 || epoch == epochs - 1 {
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
    println!("GPT training time: {:.1}s", elapsed.as_secs_f64());

    // Text generation
    println!("\n========================================");
    println!("  TEXT GENERATION");
    println!("========================================");

    let prompts = vec!["to be ", "the ", "or to "];

    for prompt in &prompts {
        let mut generated: Vec<usize> = prompt.chars().map(|c| char_to_idx[&c]).collect();

        for _ in 0..60 {
            tape.reset();
            let context_start = if generated.len() > seq_len {
                generated.len() - seq_len
            } else {
                0
            };
            let context = &generated[context_start..];
            let output = gpt.forward(&mut tape, context);

            let last_pos = context.len() - 1;
            let start = last_pos * vocab_size;
            let pred_data = &tape.tensors[output].data;
            let next_token = argmax(&pred_data[start..start + vocab_size]);
            generated.push(next_token);
        }

        let text: String = generated.iter().map(|&idx| idx_to_char[&idx]).collect();
        println!("\nPrompt: \"{}\"", prompt);
        println!("Output: \"{}\"", text);
    }

    println!("\n--- Original ---");
    println!("{}", corpus);
}
