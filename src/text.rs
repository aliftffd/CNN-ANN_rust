//! Stable hashed character n-gram features for text classification.

pub const MAX_NGRAMS: usize = 20_000;

/// Computes the FNV-1a 64-bit hash of the given bytes.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Hashes lowercase words into character 3-, 4-, and 5-grams, then whole words.
/// Words are bounded by `<` and `>`. Returns at most `MAX_NGRAMS` bucket IDs,
/// or `[0]` when the text contains no alphanumeric words.
///
/// # Panics
/// Panics if `buckets` is zero.
pub fn hashed_ngrams(text: &str, buckets: usize) -> Vec<usize> {
    assert!(buckets > 0, "buckets must be greater than zero");
    let mut ids = Vec::new();
    let lowercase = text.to_lowercase();
    for word in lowercase.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let token = format!("<{word}>");
        let chars: Vec<char> = token.chars().collect();
        for n in 3..=5 {
            for window in chars.windows(n) {
                let ngram: String = window.iter().collect();
                ids.push((fnv1a64(ngram.as_bytes()) % buckets as u64) as usize);
                if ids.len() == MAX_NGRAMS {
                    return ids;
                }
            }
        }
        ids.push((fnv1a64(token.as_bytes()) % buckets as u64) as usize);
        if ids.len() == MAX_NGRAMS {
            return ids;
        }
    }
    if ids.is_empty() {
        ids.push(0);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_pinned() {
        // Review Focus 1
        // FNV-1a % 65536 of "<hi", "hi>", "<hi>" (n=4) and "<hi>" (whole word)
        assert_eq!(hashed_ngrams("Hi", 65536), vec![45408, 56428, 25274, 25274]);
        assert_eq!(fnv1a64(b"") % 65536, 8997);
    }
    #[test]
    fn case_and_punctuation_do_not_matter() {
        assert_eq!(
            hashed_ngrams("Rapat, besok!", 65536),
            hashed_ngrams("rapat besok", 65536)
        );
    }
    #[test]
    fn ngrams_are_char_based() {
        // "<é>" is 3 chars: one 3-gram plus the whole word; no 4- or 5-grams
        assert_eq!(hashed_ngrams("é", 65536).len(), 2);
    }
    #[test]
    fn no_words_yields_bucket_zero() {
        // Review Focus 3
        for t in ["", "   \n", "!!! ---"] {
            assert_eq!(hashed_ngrams(t, 65536), vec![0]);
        }
    }
    #[test]
    fn long_text_is_capped() {
        // Review Focus 4
        let big = "kata ".repeat(512_000 / 5);
        assert_eq!(hashed_ngrams(&big, 65536).len(), MAX_NGRAMS);
    }
}
