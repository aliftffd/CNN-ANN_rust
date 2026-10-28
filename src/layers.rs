use crate::tensor::{positional_encoding, Adam, Tape};

fn xorshift(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = ((*seed >> 33) as f64) / (2f64.powi(31)) - 0.5;
    val * 0.1
}

pub struct Linear {
    pub weights: usize,
    pub bias: usize,
}

impl Linear {
    pub fn new(tape: &mut Tape, in_features: usize, out_features: usize, seed: &mut u64) -> Self {
        let scale = (2.0 / in_features as f64).sqrt();
        let weight_len = in_features * out_features;
        let weight_data: Vec<f64> = (0..weight_len).map(|_| xorshift(seed) * scale).collect();
        let weights = tape.add_tensor(weight_data, vec![in_features, out_features], true);

        let bias_data = vec![0.0; out_features];
        let bias = tape.add_tensor(bias_data, vec![out_features], true);

        Linear { weights, bias }
    }

    pub fn forward(&self, tape: &mut Tape, input: usize) -> usize {
        let z = tape.matmul(input, self.weights);
        tape.add_broadcast(z, self.bias)
    }
}

pub struct Embedding {
    pub table: usize,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl Embedding {
    pub fn new(tape: &mut Tape, vocab_size: usize, d_model: usize, seed: &mut u64) -> Self {
        let scale = (1.0 / d_model as f64).sqrt();
        let data: Vec<f64> = (0..vocab_size * d_model)
            .map(|_| xorshift(seed) * scale)
            .collect();
        let table = tape.add_tensor(data, vec![vocab_size, d_model], true);

        Embedding {
            table,
            vocab_size,
            d_model,
        }
    }

    pub fn forward(&self, tape: &mut Tape, indices: &[usize]) -> usize {
        tape.embedding(self.table, indices)
    }
}

pub struct Conv2D {
    pub kernel: usize,
    pub bias: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
}

impl Conv2D {
    pub fn new(
        tape: &mut Tape,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        seed: &mut u64,
    ) -> Self {
        let scale = (2.0 / (in_channels * kernel_h * kernel_w) as f64).sqrt();
        let kernel_len = out_channels * in_channels * kernel_h * kernel_w;
        let kernel_data: Vec<f64> = (0..kernel_len).map(|_| xorshift(seed) * scale).collect();
        let kernel = tape.add_tensor(
            kernel_data,
            vec![out_channels, in_channels, kernel_h, kernel_w],
            true,
        );

        let bias_data = vec![0.0; out_channels];
        let bias = tape.add_tensor(bias_data, vec![out_channels], true);

        Conv2D {
            kernel,
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
        }
    }

    pub fn forward(&self, tape: &mut Tape, input: usize, in_h: usize, in_w: usize) -> usize {
        tape.conv2d(
            input,
            self.kernel,
            self.bias,
            self.in_channels,
            self.out_channels,
            in_h,
            in_w,
            self.kernel_h,
            self.kernel_w,
        )
    }
}

pub struct MultiHeadAttention {
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
}

impl MultiHeadAttention {
    pub fn new(tape: &mut Tape, d_model: usize, n_heads: usize, seed: &mut u64) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;

        let w_q = Linear::new(tape, d_model, d_model, seed);
        let w_k = Linear::new(tape, d_model, d_model, seed);
        let w_v = Linear::new(tape, d_model, d_model, seed);
        let w_o = Linear::new(tape, d_model, d_model, seed);

        MultiHeadAttention {
            w_q,
            w_k,
            w_v,
            w_o,
            n_heads,
            d_model,
            d_k,
        }
    }

    pub fn forward(
        &self,
        tape: &mut Tape,
        input: usize,
        seq_len: usize,
        mask: Option<usize>,
    ) -> usize {
        // 1. Project
        let q = self.w_q.forward(tape, input);
        let k = self.w_k.forward(tape, input);
        let v = self.w_v.forward(tape, input);

        // 2. Per-head attention
        let mut head_outputs = Vec::new();
        for h in 0..self.n_heads {
            let col_start = h * self.d_k;
            let col_end = col_start + self.d_k;

            let q_h = tape.slice_cols(q, seq_len, self.d_model, col_start, col_end);
            let k_h = tape.slice_cols(k, seq_len, self.d_model, col_start, col_end);
            let v_h = tape.slice_cols(v, seq_len, self.d_model, col_start, col_end);

            let attn_h = scaled_dot_product_attention(tape, q_h, k_h, v_h, self.d_k, mask);
            head_outputs.push(attn_h);
        }

        // 3. Concatenate heads
        let concat = tape.concat_cols(&head_outputs, seq_len);

        // 4. Output projection
        self.w_o.forward(tape, concat)
    }
}

pub struct LayerNorm {
    pub gamma: usize,
    pub beta: usize,
    pub cols: usize,
}

impl LayerNorm {
    pub fn new(tape: &mut Tape, cols: usize) -> Self {
        let gamma = tape.add_tensor(vec![1.0; cols], vec![cols], true);
        let beta = tape.add_tensor(vec![0.0; cols], vec![cols], true);
        LayerNorm { gamma, beta, cols }
    }

    pub fn forward(&self, tape: &mut Tape, input: usize) -> usize {
        tape.layernorm(input, self.gamma, self.beta, self.cols, 1e-5)
    }
}

pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub ff1: Linear,
    pub ff2: Linear,
    pub norm2: LayerNorm,
    pub d_model: usize,
}

impl TransformerBlock {
    pub fn new(
        tape: &mut Tape,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        seed: &mut u64,
    ) -> Self {
        let mha = MultiHeadAttention::new(tape, d_model, n_heads, seed);
        let norm1 = LayerNorm::new(tape, d_model);
        let ff1 = Linear::new(tape, d_model, d_ff, seed);
        let ff2 = Linear::new(tape, d_ff, d_model, seed);
        let norm2 = LayerNorm::new(tape, d_model);

        TransformerBlock {
            mha,
            norm1,
            ff1,
            ff2,
            norm2,
            d_model,
        }
    }

    pub fn forward(
        &self,
        tape: &mut Tape,
        input: usize,
        seq_len: usize,
        mask: Option<usize>,
    ) -> usize {
        let attn_out = self.mha.forward(tape, input, seq_len, mask);
        let residual1 = tape.add(input, attn_out);
        let norm1_out = self.norm1.forward(tape, residual1);

        let ff_hidden = self.ff1.forward(tape, norm1_out);
        let ff_relu = tape.relu(ff_hidden);
        let ff_out = self.ff2.forward(tape, ff_relu);
        let residual2 = tape.add(norm1_out, ff_out);
        self.norm2.forward(tape, residual2)
    }
}

pub struct Transformer {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub output_proj: Linear,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub pe_idx: usize,
}

impl Transformer {
    pub fn new(
        tape: &mut Tape,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_seq_len: usize,
        seed: &mut u64,
    ) -> Self {
        let embedding = Embedding::new(tape, vocab_size, d_model, seed);

        let pe_data = positional_encoding(max_seq_len, d_model);
        let pe_idx = tape.add_tensor(pe_data, vec![max_seq_len, d_model], false);

        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(tape, d_model, n_heads, d_ff, seed));
        }

        let output_proj = Linear::new(tape, d_model, vocab_size, seed);

        Transformer {
            embedding,
            blocks,
            output_proj,
            d_model,
            max_seq_len,
            pe_idx,
        }
    }

    pub fn forward(&self, tape: &mut Tape, tokens: &[usize]) -> usize {
        let seq_len = tokens.len();

        // 1. Embed tokens → [seq_len, d_model]
        let embedded = self.embedding.forward(tape, tokens);

        // 2. Add positional encoding (slice PE to seq_len rows)
        let pe_data = tape.tensors[self.pe_idx].data[..seq_len * self.d_model].to_vec();
        let pe = tape.add_tensor(pe_data, vec![seq_len, self.d_model], false);
        let mut x = tape.add(embedded, pe);

        // 3. Pass through each transformer block (encoder — no mask)
        for block in &self.blocks {
            x = block.forward(tape, x, seq_len, None);
        }

        // 4. Output projection → [seq_len, vocab_size]
        self.output_proj.forward(tape, x)
    }
}

pub struct VisionTransformer {
    pub patch_proj: Linear,
    pub block: Vec<TransformerBlock>,
    pub classifier: Linear,
    pub d_model: usize,
    pub seq_len: usize,
    pub pe_idx: usize,
}

impl VisionTransformer {
    pub fn new(
        tape: &mut Tape,
        patch_size: usize,
        seq_len: usize,
        d_model: usize,
        n_head: usize,
        d_ff: usize,
        n_layers: usize,
        num_classes: usize,
        seed: &mut u64,
    ) -> Self {
        let patch_proj = Linear::new(tape, patch_size, d_model, seed);

        let pe_data = positional_encoding(seq_len, d_model);
        let pe_idx = tape.add_tensor(pe_data, vec![seq_len, d_model], false);

        let mut block = Vec::new();
        for _ in 0..n_layers {
            block.push(TransformerBlock::new(tape, d_model, n_head, d_ff, seed));
        }

        let classifier = Linear::new(tape, d_model, num_classes, seed);

        VisionTransformer {
            patch_proj,
            block,
            classifier,
            d_model,
            seq_len,
            pe_idx,
        }
    }

    pub fn forward(&self, tape: &mut Tape, image: &[f64]) -> usize {
        let x = tape.add_tensor(image.to_vec(), vec![self.seq_len, 28], false);

        let projected = self.patch_proj.forward(tape, x);

        let ped_data = tape.tensors[self.pe_idx].data[..self.seq_len * self.d_model].to_vec();
        let pe = tape.add_tensor(ped_data, vec![self.seq_len, self.d_model], false);
        let mut h = tape.add(projected, pe);

        for block in &self.block {
            h = block.forward(tape, h, self.seq_len, None);
        }

        let pooled = tape.mean_pool(h, self.seq_len, self.d_model);

        self.classifier.forward(tape, pooled)
    }
}

pub fn scaled_dot_product_attention(
    tape: &mut Tape,
    q: usize,
    k: usize,
    v: usize,
    d_k: usize,
    mask: Option<usize>,
) -> usize {
    let k_t = tape.transpose(k, 0, 1);
    let scores = tape.matmul(q, k_t);
    let scale = 1.0 / (d_k as f64).sqrt();
    let scaled = tape.scalar_mul(scores, scale);

    let masked = match mask {
        Some(mask_idx) => tape.add(scaled, mask_idx),
        None => scaled,
    };

    let seq_len = tape.tensors[q].shape[0];
    let weights = tape.softmax(masked, seq_len);
    tape.matmul(weights, v)
}

/// Generates a causal (lower-triangular) mask for autoregressive attention.
/// Future positions get -1e9 (effectively -inf after softmax), past/current get 0.
pub fn causal_mask(seq_len: usize) -> Vec<f64> {
    let mut mask = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask[i * seq_len + j] = -1e9;
            }
        }
    }
    mask
}

pub struct GPT {
    pub embedding: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub output_proj: Linear,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub pe_idx: usize,
    pub vocab_size: usize,
}

impl GPT {
    pub fn new(
        tape: &mut Tape,
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        n_layers: usize,
        max_seq_len: usize,
        seed: &mut u64,
    ) -> Self {
        let embedding = Embedding::new(tape, vocab_size, d_model, seed);
        let pe_data = positional_encoding(max_seq_len, d_model);
        let pe_idx = tape.add_tensor(pe_data, vec![max_seq_len, d_model], false);

        let mut blocks = Vec::new();
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(tape, d_model, n_heads, d_ff, seed));
        }

        let output_proj = Linear::new(tape, d_model, vocab_size, seed);

        GPT {
            embedding,
            blocks,
            output_proj,
            d_model,
            max_seq_len,
            pe_idx,
            vocab_size,
        }
    }

    pub fn forward(&self, tape: &mut Tape, tokens: &[usize]) -> usize {
        let seq_len = tokens.len();

        // 1. Embed tokens
        let embedded = self.embedding.forward(tape, tokens);

        // 2. Add positional encoding
        let pe_data = tape.tensors[self.pe_idx].data[..seq_len * self.d_model].to_vec();
        let pe = tape.add_tensor(pe_data, vec![seq_len, self.d_model], false);
        let mut x = tape.add(embedded, pe);

        // 3. Create causal mask and add to tape
        let mask_data = causal_mask(seq_len);
        let mask = tape.add_tensor(mask_data, vec![seq_len, seq_len], false);

        // 4. Pass through decoder blocks WITH mask
        for block in &self.blocks {
            x = block.forward(tape, x, seq_len, Some(mask));
        }

        // 5. Project to vocab
        self.output_proj.forward(tape, x)
    }
}

// ponytail: one tape per model; batch on one tape, Adam step per batch, not per example.
/// Bag-of-n-grams MLP for text classification:
/// ids -> Embedding -> MeanPool -> Linear -> ReLU -> Linear(n_classes).
pub struct TextMlp {
    pub emb: Embedding,
    pub hidden: Linear,
    pub out: Linear,
    pub n_classes: usize,
}

impl TextMlp {
    pub fn new(
        tape: &mut Tape,
        buckets: usize,
        dim: usize,
        hidden: usize,
        n_classes: usize,
        seed: &mut u64,
    ) -> Self {
        TextMlp {
            emb: Embedding::new(tape, buckets, dim, seed),
            hidden: Linear::new(tape, dim, hidden, seed),
            out: Linear::new(tape, hidden, n_classes, seed),
            n_classes,
        }
    }

    /// Returns logits with shape [1, n_classes].
    pub fn forward(&self, tape: &mut Tape, ids: &[usize]) -> usize {
        let embedded = self.emb.forward(tape, ids);
        let pooled = tape.mean_pool(embedded, ids.len(), self.emb.d_model);
        let h = self.hidden.forward(tape, pooled);
        let h = tape.relu(h);
        self.out.forward(tape, h)
    }

    /// One Adam step on the mean cross-entropy of `batch`; returns that loss.
    pub fn train_batch(&self, tape: &mut Tape, adam: &mut Adam, batch: &[(Vec<usize>, usize)]) -> f64 {
        let mut total: Option<usize> = None;
        for (ids, label) in batch {
            let logits = self.forward(tape, ids);
            let mut one_hot = vec![0.0; self.n_classes];
            one_hot[*label] = 1.0;
            let target = tape.add_tensor(one_hot, vec![1, self.n_classes], false);
            let loss = tape.softmax_cross_entropy(logits, target, self.n_classes);
            total = Some(match total {
                None => loss,
                Some(acc) => tape.add(acc, loss),
            });
        }
        let total = total.expect("train_batch needs at least one example");
        let mean = tape.scalar_mul(total, 1.0 / batch.len() as f64);
        let loss = tape.tensors[mean].data[0];
        tape.backward(mean);
        adam.step(tape);
        tape.reset();
        loss
    }

    /// Class probabilities; resets the tape so repeated calls don't grow it.
    pub fn predict(&self, tape: &mut Tape, ids: &[usize]) -> Vec<f64> {
        let logits = self.forward(tape, ids);
        let probs = tape.softmax(logits, self.n_classes);
        let out = tape.tensors[probs].data.clone();
        tape.reset();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Adam;
    use crate::text::hashed_ngrams;

    const B: usize = 4096;

    fn model() -> (Tape, TextMlp) {
        let mut t = Tape::new();
        let mut s = 42;
        let m = TextMlp::new(&mut t, B, 16, 16, 2, &mut s);
        t.freeze_params();
        (t, m)
    }

    fn argmax(p: &[f64]) -> usize {
        (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()
    }

    #[test]
    fn learns_toy_mixed_language_data() {
        let data = [
            ("rapat tim besok jam 9", 0),
            ("meeting agenda for sprint review", 0),
            ("beli telur dan susu", 1),
            ("grocery list milk eggs bread", 1),
        ];
        let batch: Vec<_> = data.iter().map(|(t, y)| (hashed_ngrams(t, B), *y)).collect();
        let (mut t, m) = model();
        let mut adam = Adam::new(&t, 0.01);
        let first = m.train_batch(&mut t, &mut adam, &batch);
        let mut last = first;
        for _ in 0..200 {
            last = m.train_batch(&mut t, &mut adam, &batch);
        }
        assert!(last < first * 0.1, "{first} -> {last}");
        for (ids, y) in &batch {
            assert_eq!(argmax(&m.predict(&mut t, ids)), *y);
        }
    }

    #[test]
    fn predict_does_not_grow_tape() {
        let (mut t, m) = model();
        let ids = hashed_ngrams("catatan", B);
        for _ in 0..3 {
            m.predict(&mut t, &ids);
        }
        assert_eq!(t.tensors.len(), t.param_count);
    }

    #[test]
    fn predict_on_empty_text_is_finite() {
        let (mut t, m) = model();
        let p = m.predict(&mut t, &hashed_ngrams("", B));
        assert!(p.iter().all(|x| x.is_finite()) && (p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn gradcheck_embedding_row() {
        let (mut t, m) = model();
        let ids = hashed_ngrams("halo", B);
        let loss = |t: &mut Tape| {
            let logits = m.forward(t, &ids);
            let target = t.add_tensor(vec![0.0, 1.0], vec![1, 2], false);
            t.softmax_cross_entropy(logits, target, 2)
        };
        let o = loss(&mut t);
        t.backward(o);
        let k = ids[0] * 16;
        let analytic = t.tensors[m.emb.table].grad[k];
        t.reset();

        let eps = 1e-5;
        t.tensors[m.emb.table].data[k] += eps;
        let o = loss(&mut t);
        let up = t.tensors[o].data[0];
        t.reset();
        t.tensors[m.emb.table].data[k] -= 2.0 * eps;
        let o = loss(&mut t);
        let down = t.tensors[o].data[0];
        t.reset();

        assert!((analytic - (up - down) / (2.0 * eps)).abs() < 1e-6, "{analytic}");
    }
}
