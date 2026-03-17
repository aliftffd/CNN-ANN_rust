use crate::tensor::Tape;

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

