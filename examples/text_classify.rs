//! Classify short Indonesian/English notes as kerja, belanja, or kuliah.
//!
//! Architecture:
//!   ids = hashed_ngrams(text) -> Embedding(65536,32) -> MeanPool
//!       -> Linear(32) -> ReLU -> Linear(n_classes) -> Softmax+CE
//!
//! Uses Adam (lr 0.01), batches of 8, and 30 epochs on 18 training notes.
//! Six held-out notes test accuracy and exact predictions after saving/loading.
//!
//! Run: cargo run --release --example text_classify

use rust_ann::layers::TextMlp;
use rust_ann::tensor::{Adam, Tape};
use rust_ann::text::hashed_ngrams;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::time::Instant;

const BUCKETS: usize = 65_536;
const DIM: usize = 32;
const HIDDEN: usize = 32;
const LEARNING_RATE: f64 = 0.01;
const BATCH_SIZE: usize = 8;
const EPOCHS: usize = 30;
const CATEGORIES: [&str; 3] = ["kerja", "belanja", "kuliah"];

fn argmax(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..probabilities.len() {
        if probabilities[i] > probabilities[best] {
            best = i;
        }
    }
    best
}

fn main() -> io::Result<()> {
    let start = Instant::now();

    // Interleave categories so each batch contains a mix; labels follow CATEGORIES.
    let train_notes = [
        ("Rapat tim besok jam sembilan, prepare sprint update", 0),
        ("Beli telur dan susu di supermarket sepulang nanti", 1),
        ("Kerjakan tugas aljabar sebelum kuliah hari Senin", 2),
        ("Send the client proposal before Friday's meeting", 0),
        ("Grocery list: bread, eggs, milk and apples", 1),
        ("Study calculus chapter three for the exam", 2),
        ("Review laporan proyek, lalu email ke manager", 0),
        ("Stok beras habis, add rice to the shopping list", 1),
        ("Review catatan kuliah statistik untuk quiz besok", 2),
        ("Fix bug login sebelum deploy ke production", 0),
        ("Buy detergent and toothpaste, cek promo toko dulu", 1),
        ("Finish physics homework lalu upload ke portal kampus", 2),
        ("Siapkan agenda meeting dengan klien sore ini", 0),
        ("Belanja sayur, ayam, dan bawang di pasar pagi", 1),
        ("Baca materi ujian dan rangkum bab probabilitas", 2),
        ("Update project deadline dan kirim report ke tim", 0),
        ("Pick up coffee and sugar, jangan lupa beli sabun", 1),
        ("Read lecture slides sebelum diskusi dengan dosen", 2),
    ];
    // These six notes never enter train_batch (two per category).
    let test_notes = [
        ("Email laporan proyek ke klien before the meeting", 0),
        ("Besok sprint review, siapkan update untuk tim", 0),
        ("Beli susu, bread dan telur buat sarapan", 1),
        ("Shopping list: beras, sabun, and coffee", 1),
        ("Study catatan kuliah untuk ujian statistik", 2),
        ("Finish tugas calculus dan baca lecture slides", 2),
    ];
    let train: Vec<_> = train_notes
        .iter()
        .map(|(text, label)| (hashed_ngrams(text, BUCKETS), *label))
        .collect();

    let mut tape = Tape::new();
    let mut seed = 42;
    let model = TextMlp::new(&mut tape, BUCKETS, DIM, HIDDEN, CATEGORIES.len(), &mut seed);
    tape.freeze_params();
    let mut adam = Adam::new(&tape, LEARNING_RATE);

    println!(
        "Text classification: {} train / {} test notes",
        train.len(),
        test_notes.len()
    );
    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        // Keep the final partial batch; weight its mean by its actual size.
        for batch in train.chunks(BATCH_SIZE) {
            total_loss += model.train_batch(&mut tape, &mut adam, batch) * batch.len() as f64;
        }
        println!(
            "Epoch {epoch:2}/{EPOCHS}: loss = {:.6}",
            total_loss / train.len() as f64
        );
    }

    let mut correct = 0;
    let mut predictions = Vec::new();
    for (text, label) in &test_notes {
        let ids = hashed_ngrams(text, BUCKETS);
        let probabilities = model.predict(&mut tape, &ids);
        let predicted = argmax(&probabilities);
        if predicted == *label {
            correct += 1;
        }
        println!(
            "{} (expected {}): {text}",
            CATEGORIES[predicted], CATEGORIES[*label]
        );
        predictions.push(probabilities);
    }
    println!(
        "Test accuracy: {:.2}% ({correct}/{})",
        100.0 * correct as f64 / test_notes.len() as f64,
        test_notes.len()
    );

    fs::create_dir_all("target")?;
    let path = "target/text_mlp.bin";
    {
        let mut writer = BufWriter::new(File::create(path)?);
        tape.save_params(&mut writer)?;
        writer.flush()?;
    }
    let mut loaded_tape = Tape::new();
    let mut loaded_seed = 99;
    let loaded_model = TextMlp::new(
        &mut loaded_tape,
        BUCKETS,
        DIM,
        HIDDEN,
        CATEGORIES.len(),
        &mut loaded_seed,
    );
    loaded_tape.freeze_params();
    loaded_tape.load_params(&mut BufReader::new(File::open(path)?))?;
    for ((text, _), expected) in test_notes.iter().zip(&predictions) {
        let actual = loaded_model.predict(&mut loaded_tape, &hashed_ngrams(text, BUCKETS));
        assert_eq!(&actual, expected, "prediction changed after reload: {text}");
    }
    println!("Saved {path}; reloaded predictions are identical on all six test notes.");
    println!("Example wall time: {:.2}s", start.elapsed().as_secs_f64());
    Ok(())
}
