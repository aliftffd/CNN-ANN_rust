use crate::tensor::{positional_encoding, Tape};

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
        let out = tape.add_broadcast(z, self.bias);
        out
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

    pub fn forward(&self, tape: &mut Tape, input: usize, seq_len: usize) -> usize {
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

            let attn_h = scaled_dot_product_attention(tape, q_h, k_h, v_h, self.d_k);
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

    pub fn forward(&self, tape: &mut Tape, input: usize, seq_len: usize) -> usize {
        // Multi-head attention + residual + layernorm
        let attn_out = self.mha.forward(tape, input, seq_len);
        let residual1 = tape.add(input, attn_out);
        let norm1_out = self.norm1.forward(tape, residual1);

        // FFN + residual + layernorm
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

        // 3. Pass through each transformer block
        for block in &self.blocks {
            x = block.forward(tape, x, seq_len);
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
            h = block.forward(tape, h, self.seq_len);
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
) -> usize {
    let k_t = tape.transpose(k, 0, 1);
    let scores = tape.matmul(q, k_t);
    let scale = 1.0 / (d_k as f64).sqrt();
    let scaled = tape.scalar_mul(scores, scale);
    let seq_len = tape.tensors[q].shape[0];
    let weights = tape.softmax(scaled, seq_len);
    tape.matmul(weights, v)
}
