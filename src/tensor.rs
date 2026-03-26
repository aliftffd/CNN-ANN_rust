#[derive(Debug, Clone)]
pub enum Op {
    None,
    Add(usize, usize),
    Mul(usize, usize),
    MatMul(usize, usize),
    AddBroadcast(usize, usize),
    ReLU(usize),
    MSE(usize, usize),
    Conv2D {
        input: usize,
        kernel: usize,
        bias: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        in_h: usize,
        in_w: usize,
    },
    MaxPool2D {
        input: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        pool_size: usize,
    },
    Flatten {
        input: usize,
        channels: usize,
        h: usize,
        w: usize,
    },
    Softmax {
        input: usize,
        cols: usize,
    },
    ScalarMul {
        input: usize,
        scalar: f64,
    },
    Reshape {
        input: usize,
    },
    Transpose {
        input: usize,
        dim0: usize,
        dim1: usize,
    },

    SoftmaxCrossEntropy {
        input: usize,
        target: usize,
        cols: usize,
    },

    LayerNorm {
        input: usize,
        gamma: usize,
        beta: usize,
        cols: usize,
        eps: f64,
    },

    Embedding {
        table: usize,
        indices: Vec<usize>,
    },

    SliceCols {
        input: usize,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_end: usize,
    },

    ConcatCols {
        inputs: Vec<usize>,
        rows: usize,
    },

    MeanPool {
        input: usize,
        rows: usize,
        cols: usize,
    },
}

pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub grad: Vec<f64>,
    pub requires_grad: bool,
    pub op: Op,
}

pub struct Tape {
    pub tensors: Vec<Tensor>,
    pub param_count: usize,
}

pub fn raw_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

pub fn raw_transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = data[i * cols + j];
        }
    }
    transposed
}

pub fn positional_encoding(max_len: usize, d_model: usize) -> Vec<f64> {
    let mut pe = Vec::with_capacity(max_len * d_model);
    for pos in 0..max_len {
        for i in 0..d_model {
            let div_term = (10000.0_f64).powf(((i / 2) * 2) as f64 / d_model as f64);
            let val = if i % 2 == 0 {
                (pos as f64 / div_term).sin()
            } else {
                (pos as f64 / div_term).cos()
            };
            pe.push(val);
        }
    }
    pe
}

pub fn sgd_step(tape: &mut Tape, learning_rate: f64) {
    for tensor in tape.tensors.iter_mut() {
        if tensor.requires_grad {
            for (d, g) in tensor.data.iter_mut().zip(tensor.grad.iter_mut()) {
                *d -= learning_rate * *g;
                *g = 0.0;
            }
        }
    }
}

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub t: usize,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(tape: &Tape, lr: f64) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();

        for tensor in &tape.tensors {
            if tensor.requires_grad {
                m.push(vec![0.0; tensor.data.len()]);
                v.push(vec![0.0; tensor.data.len()]);
            } else {
                m.push(Vec::new());
                v.push(Vec::new());
            }
        }

        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m,
            v,
        }
    }

    pub fn step(&mut self, tape: &mut Tape) {
        self.t += 1;

        for (i, tensor) in tape.tensors[..tape.param_count].iter_mut().enumerate() {
            if !tensor.requires_grad {
                continue;
            }

            for (j, (d, g)) in tensor
                .data
                .iter_mut()
                .zip(tensor.grad.iter_mut())
                .enumerate()
            {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * *g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * *g * *g;

                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));

                *d -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                *g = 0.0;
            }
        }
    }
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            tensors: Vec::new(),
            param_count: 0,
        }
    }

    pub fn add_tensor(&mut self, data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> usize {
        let grad = if requires_grad {
            vec![0.0; data.len()]
        } else {
            Vec::new()
        };

        let tensor = Tensor {
            data,
            shape,
            grad,
            requires_grad,
            op: Op::None,
        };
        self.tensors.push(tensor);
        self.tensors.len() - 1
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let tensor_a = &self.tensors[a];
        let tensor_b = &self.tensors[b];

        // debug
        assert_eq!(
            tensor_a.shape, tensor_b.shape,
            "Shape must match for element-wise mul"
        );

        let result_data: Vec<f64> = tensor_a
            .data
            .iter()
            .zip(&tensor_b.data)
            .map(|(x, y)| x * y)
            .collect();

        let requires_grad = tensor_a.requires_grad || tensor_b.requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: tensor_a.shape.clone(),
            grad,
            requires_grad,
            op: Op::Mul(a, b),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let tensor_a = &self.tensors[a];
        let tensor_b = &self.tensors[b];

        // debug
        assert_eq!(
            tensor_a.shape, tensor_b.shape,
            "Shape must match for element-wise add"
        );

        let result_data: Vec<f64> = tensor_a
            .data
            .iter()
            .zip(&tensor_b.data)
            .map(|(x, y)| x + y)
            .collect();

        let requires_grad = tensor_a.requires_grad || tensor_b.requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: tensor_a.shape.clone(),
            grad,
            requires_grad,
            op: Op::Add(a, b),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let tensor_a = &self.tensors[a];
        let tensor_b = &self.tensors[b];

        assert_eq!(tensor_a.shape.len(), 2, "Ts A Must be 2D");
        assert_eq!(tensor_b.shape.len(), 2, "Ts B must be 2D");

        let m = tensor_a.shape[0];
        let k = tensor_a.shape[1];
        let n = tensor_b.shape[1];

        assert_eq!(
            k, tensor_b.shape[0],
            "inner dimensions must match for matrix multiplication"
        );
        assert_eq!(tensor_a.data.len(), m * k);
        assert_eq!(tensor_b.data.len(), k * n);

        let mut result_data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a_idx = i * k + p;
                    let b_idx = p * n + j;
                    sum += tensor_a.data[a_idx] * tensor_b.data[b_idx];
                }
                result_data[i * n + j] = sum;
            }
        }

        let requires_grad = tensor_a.requires_grad || tensor_b.requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: vec![m, n],
            grad,
            requires_grad,
            op: Op::MatMul(a, b),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn add_broadcast(&mut self, matrix_idx: usize, bias_idx: usize) -> usize {
        let matrix = &self.tensors[matrix_idx];
        let bias = &self.tensors[bias_idx];

        assert_eq!(matrix.shape.len(), 2, "matrix must be 2D");
        assert_eq!(bias.shape.len(), 1, "bias must be 1D");
        assert_eq!(
            matrix.shape[1], bias.shape[0],
            "bias length must match matrix columns"
        );

        let m = matrix.shape[0];
        let n = matrix.shape[1];

        let mut result_data = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                result_data[row * n + col] = matrix.data[row * n + col] + bias.data[col];
            }
        }

        let requires_grad = matrix.requires_grad || bias.requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: matrix.shape.clone(),
            grad,
            requires_grad,
            op: Op::AddBroadcast(matrix_idx, bias_idx),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn relu(&mut self, input_idx: usize) -> usize {
        let input = &self.tensors[input_idx];
        let result_data: Vec<f64> = input.data.iter().map(|&x| x.max(0.0)).collect();

        let requires_grad = input.requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: input.shape.clone(),
            grad,
            requires_grad,
            op: Op::ReLU(input_idx),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn mse_loss(&mut self, pred_idx: usize, target_idx: usize) -> usize {
        let pred = &self.tensors[pred_idx];
        let target = &self.tensors[target_idx];

        assert_eq!(
            pred.shape, target.shape,
            "prediction and target must match for MSE"
        );

        let n = pred.data.len() as f64;
        let sum_sq: f64 = pred
            .data
            .iter()
            .zip(&target.data)
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        let loss = sum_sq / n;

        let requires_grad = pred.requires_grad;
        let grad = if requires_grad {
            vec![0.0; 1]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: vec![loss],
            shape: vec![1],
            grad,
            requires_grad,
            op: Op::MSE(pred_idx, target_idx),
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn freeze_params(&mut self) {
        self.param_count = self.tensors.len();
    }

    pub fn reset(&mut self) {
        self.tensors.truncate(self.param_count);
        for i in 0..self.param_count {
            if self.tensors[i].requires_grad {
                for g in self.tensors[i].grad.iter_mut() {
                    *g = 0.0
                }
            }
        }
    }

    pub fn conv2d(
        &mut self,
        input_idx: usize,
        kernel_idx: usize,
        bias_idx: usize,
        in_channels: usize,
        out_channels: usize,
        in_h: usize,
        in_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> usize {
        // compute output spatioal dimensions (with no padding and stride = 1.0)
        let out_h = in_h - kernel_h + 1;
        let out_w = in_w - kernel_w + 1;
        assert!(out_h > 0 && out_w > 0, "Kernel too large for input");

        // get data slices from the stored tensor
        let input_data = &self.tensors[input_idx].data;
        let kernel_data = &self.tensors[kernel_idx].data;
        let bias_data = &self.tensors[bias_idx].data;

        // prepare output buffer
        let out_len = out_channels * out_h * out_w;
        let mut result_data = vec![0.0; out_len];

        for oc in 0..out_channels {
            for r in 0..out_h {
                for c in 0..out_w {
                    let mut sum = 0.0;
                    for ic in 0..in_channels {
                        for kr in 0..kernel_h {
                            for kc in 0..kernel_w {
                                let input_idx_flat = ic * in_h * in_w + (r + kr) * in_w + (c + kc);
                                let kernel_idx_flat = oc * in_channels * kernel_h * kernel_w
                                    + ic * kernel_h * kernel_w
                                    + kr * kernel_w
                                    + kc;
                                sum += input_data[input_idx_flat] * kernel_data[kernel_idx_flat];
                            }
                        }
                    }
                    sum += bias_data[oc];
                    let out_idx_flat = oc * out_h * out_w + r * out_w + c;
                    result_data[out_idx_flat] = sum;
                }
            }
        }

        let requires_grad = self.tensors[input_idx].requires_grad
            || self.tensors[kernel_idx].requires_grad
            || self.tensors[bias_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_data = Tensor {
            data: result_data,
            shape: vec![out_channels, out_h, out_w],
            grad,
            requires_grad,
            op: Op::Conv2D {
                input: input_idx,
                kernel: kernel_idx,
                bias: bias_idx,
                in_channels,
                out_channels,
                kernel_h,
                kernel_w,
                in_h,
                in_w,
            },
        };
        self.tensors.push(result_data);
        self.tensors.len() - 1
    }

    pub fn max_pool2d(
        &mut self,
        input_idx: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        pool_size: usize,
    ) -> usize {
        let out_h = in_h / pool_size;
        let out_w = in_w / pool_size;

        let input_data = &self.tensors[input_idx].data;
        let mut result_data = vec![0.0; channels * out_h * out_w];

        for ch in 0..channels {
            for r in 0..out_h {
                for c in 0..out_w {
                    let mut max_val = f64::NEG_INFINITY;
                    for pr in 0..pool_size {
                        for pc in 0..pool_size {
                            let in_r = r * pool_size + pr;
                            let in_c = c * pool_size + pc;
                            let idx = ch * in_h * in_w + in_r * in_w + in_c;
                            if input_data[idx] > max_val {
                                max_val = input_data[idx];
                            }
                        }
                    }
                    let out_idx = ch * out_h * out_w + r * out_w + c;
                    result_data[out_idx] = max_val;
                }
            }
        }

        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_data = Tensor {
            data: result_data,
            shape: vec![channels, out_h, out_w],
            grad,
            requires_grad,
            op: Op::MaxPool2D {
                input: input_idx,
                channels,
                in_h,
                in_w,
                pool_size,
            },
        };
        self.tensors.push(result_data);
        self.tensors.len() - 1
    }

    pub fn flatten_2d(&mut self, input_idx: usize, channels: usize, h: usize, w: usize) -> usize {
        let input_data = self.tensors[input_idx].data.clone();
        let requires_grad = self.tensors[input_idx].requires_grad;
        let len = channels * h * w;
        let grad = if requires_grad {
            vec![0.0; len]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: input_data,
            shape: vec![1, len],
            grad,
            requires_grad,
            op: Op::Flatten {
                input: input_idx,
                channels,
                h,
                w,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn softmax(&mut self, input_idx: usize, cols: usize) -> usize {
        // get input data
        let input_data = &self.tensors[input_idx].data;
        let total_len = input_data.len();

        let rows = total_len / cols;
        assert_eq!(rows * cols, total_len, "Input length must be multiple cols");

        // prepare output buffer
        let mut output_data = vec![0.0; total_len];

        for r in 0..rows {
            let row_start = r * cols;
            let row = &input_data[row_start..row_start + cols];

            let mut max_val = row[0];
            for &x in row.iter().skip(1) {
                if x > max_val {
                    max_val = x;
                }
            }

            let mut sum = 0.0;
            let mut exps = vec![0.0; cols];
            for (i, &x) in row.iter().enumerate() {
                let e = (x - max_val).exp();
                exps[i] = e;
                sum += e;
            }

            for i in 0..cols {
                output_data[row_start + i] = exps[i] / sum;
            }
        }

        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; total_len]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: output_data,
            shape: self.tensors[input_idx].shape.clone(),
            grad,
            requires_grad,
            op: Op::Softmax {
                input: input_idx,
                cols,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn scalar_mul(&mut self, input_idx: usize, scalar: f64) -> usize {
        // get input data
        let input_data = &self.tensors[input_idx].data;
        //mulyiple every element of scalar
        let result_data: Vec<f64> = input_data.iter().map(|&x| x * scalar).collect();

        // create tensor
        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: self.tensors[input_idx].shape.clone(),
            grad,
            requires_grad,
            op: Op::ScalarMul {
                input: input_idx,
                scalar,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn reshape(&mut self, input_idx: usize, new_shape: Vec<usize>) -> usize {
        let old_len: usize = self.tensors[input_idx].data.len();
        let new_len: usize = new_shape.iter().product();
        assert_eq!(old_len, new_len, "Reshape: total elemnts must match");

        let data = self.tensors[input_idx].data.clone();
        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; old_len]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data,
            shape: new_shape,
            grad,
            requires_grad,
            op: Op::Reshape { input: input_idx },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn transpose(&mut self, input_idx: usize, dim0: usize, dim1: usize) -> usize {
        let old_shape = self.tensors[input_idx].shape.clone();
        let ndim = old_shape.len();
        assert!(dim0 < ndim && dim1 < ndim, "dims out of range");

        // new shape swap dim0 and dim1
        let mut new_shape = old_shape.clone();
        new_shape.swap(dim0, dim1);

        let total_len = old_shape.iter().product::<usize>();
        let input_data = &self.tensors[input_idx].data;
        let mut result_data = vec![0.0; total_len];

        // for each element compute coords, swap, compute new flat index
        for old_flat in 0..total_len {
            // conver flat to coordinates using old_shape
            let mut coords = vec![0usize; ndim];
            let mut remaining = old_flat;
            for d in 0..ndim {
                let stride: usize = old_shape[d + 1..].iter().product();
                coords[d] = remaining / stride;
                remaining %= stride;
            }

            // swap coordinates
            coords.swap(dim0, dim1);

            // conver coordinates to flat using new_shape
            let mut new_flat = 0;
            for d in 0..ndim {
                let stride: usize = new_shape[d + 1..].iter().product();
                new_flat += coords[d] * stride;
            }

            result_data[new_flat] = input_data[old_flat];
        }

        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; total_len]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: new_shape,
            grad,
            requires_grad,
            op: Op::Transpose {
                input: input_idx,
                dim0,
                dim1,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn softmax_cross_entropy(
        &mut self,
        input_idx: usize,
        target_idx: usize,
        cols: usize,
    ) -> usize {
        let input_data = &self.tensors[input_idx].data;
        let target_data = &self.tensors[target_idx].data;
        let total_len = input_data.len();
        let rows = total_len / cols;

        // compute softmax
        let mut softmax_out = vec![0.0; total_len];
        for r in 0..rows {
            let start = r * cols;
            let row = &input_data[start..start + cols];

            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut sum = 0.0;
            for i in 0..cols {
                let e = (row[i] - max_val).exp();
                softmax_out[start + i] = e;
                sum += e;
            }

            for i in 0..cols {
                softmax_out[start + i] /= sum;
            }
        }

        let mut loss = 0.0; // compute cross-entropy loss
        for i in 0..total_len {
            loss -= target_data[i] * (softmax_out[i] + 1e-8).ln();
        }
        loss /= rows as f64;

        // store the softmax outuput as a tensor
        let sm_tensor = Tensor {
            data: softmax_out,
            shape: self.tensors[input_idx].shape.clone(),
            grad: Vec::new(),
            requires_grad: false,
            op: Op::None,
        };
        self.tensors.push(sm_tensor);
        let sm_idx = self.tensors.len() - 1;

        // create loss tensor
        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; 1]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: vec![loss],
            shape: vec![1],
            grad,
            requires_grad,
            op: Op::SoftmaxCrossEntropy {
                input: input_idx,
                target: target_idx,
                cols,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn layernorm(
        &mut self,
        input_idx: usize,
        gamma_idx: usize,
        beta_idx: usize,
        cols: usize,
        eps: f64,
    ) -> usize {
        let input_data = &self.tensors[input_idx].data;
        let gamma_data = &self.tensors[gamma_idx].data;
        let beta_data = &self.tensors[beta_idx].data;
        let total_len = input_data.len();
        let rows = total_len / cols;

        let mut output_data = vec![0.0; total_len];
        let mut norm_data = vec![0.0; total_len];

        for r in 0..rows {
            let start = r * cols;
            // calculate mean
            let mut mean = 0.0;
            for c in 0..cols {
                mean += input_data[start + c];
            }
            mean /= cols as f64;

            // calculate variance
            let mut var = 0.0;
            for c in 0..cols {
                let diff = input_data[start + c] - mean;
                var += diff * diff;
            }

            var /= cols as f64;

            let inv_std = 1.0 / (var + eps).sqrt();

            // normalize, scale shift
            for c in 0..cols {
                let norm = (input_data[start + c] - mean) * inv_std;
                norm_data[start + c] = norm;
                output_data[start + c] = gamma_data[c] * norm + beta_data[c];
            }
        }

        // store norma data as a hidden tensor
        let norm_tensor = Tensor {
            data: norm_data,
            shape: self.tensors[input_idx].shape.clone(),
            grad: Vec::new(),
            requires_grad: false,
            op: Op::None,
        };
        self.tensors.push(norm_tensor);

        // create output tensor
        let requires_grad = self.tensors[input_idx].requires_grad
            || self.tensors[gamma_idx].requires_grad
            || self.tensors[beta_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; total_len]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: output_data,
            shape: self.tensors[input_idx].shape.clone(),
            grad,
            requires_grad,
            op: Op::LayerNorm {
                input: input_idx,
                gamma: gamma_idx,
                beta: beta_idx,
                cols,
                eps,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn embedding(&mut self, table_idx: usize, indices: &[usize]) -> usize {
        let table_data = &self.tensors[table_idx].data;
        let table_shape = &self.tensors[table_idx].shape;
        let vocab_size = table_shape[0];
        let d_model = table_shape[1];
        let seq_len = indices.len();

        // Look up each row
        let mut result_data = Vec::with_capacity(seq_len * d_model);
        for &idx in indices {
            assert!(
                idx < vocab_size,
                "Token index {} out of vocab range {}",
                idx,
                vocab_size
            );
            let start = idx * d_model;
            result_data.extend_from_slice(&table_data[start..start + d_model]);
        }

        let requires_grad = self.tensors[table_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; seq_len * d_model]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: vec![seq_len, d_model],
            grad,
            requires_grad,
            op: Op::Embedding {
                table: table_idx,
                indices: indices.to_vec(),
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn slice_cols(
        &mut self,
        input_idx: usize,
        rows: usize,
        total_cols: usize,
        col_start: usize,
        col_end: usize,
    ) -> usize {
        let input_data = &self.tensors[input_idx].data;
        let slice_cols = col_end - col_start;
        let mut result_data = Vec::with_capacity(rows * slice_cols);

        for r in 0..rows {
            for c in col_start..col_end {
                result_data.push(input_data[r * total_cols + c]);
            }
        }

        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: vec![rows, slice_cols],
            grad,
            requires_grad,
            op: Op::SliceCols {
                input: input_idx,
                rows,
                total_cols,
                col_start,
                col_end,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn concat_cols(&mut self, input_indices: &[usize], rows: usize) -> usize {
        let total_cols: usize = input_indices
            .iter()
            .map(|&idx| self.tensors[idx].shape[1])
            .sum();

        let mut result_data = Vec::with_capacity(rows * total_cols);
        for r in 0..rows {
            for &idx in input_indices {
                let cols = self.tensors[idx].shape[1];
                let start = r * cols;
                result_data.extend_from_slice(&self.tensors[idx].data[start..start + cols]);
            }
        }

        let requires_grad = input_indices
            .iter()
            .any(|&idx| self.tensors[idx].requires_grad);
        let grad = if requires_grad {
            vec![0.0; result_data.len()]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: vec![rows, total_cols],
            grad,
            requires_grad,
            op: Op::ConcatCols {
                inputs: input_indices.to_vec(),
                rows,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn mean_pool(&mut self, input_idx: usize, rows: usize, cols: usize) -> usize {
        let input_data = &self.tensors[input_idx].data;
        let mut result_data = vec![0.0; cols];

        for r in 0..rows {
            for c in 0..cols {
                result_data[c] += input_data[r * cols + c];
            }
        }
        for c in 0..cols {
            result_data[c] /= rows as f64;
        }

        let requires_grad = self.tensors[input_idx].requires_grad;
        let grad = if requires_grad {
            vec![0.0; cols]
        } else {
            Vec::new()
        };

        let result_tensor = Tensor {
            data: result_data,
            shape: vec![1, cols],
            grad,
            requires_grad,
            op: Op::MeanPool {
                input: input_idx,
                rows,
                cols,
            },
        };
        self.tensors.push(result_tensor);
        self.tensors.len() - 1
    }

    pub fn backward(&mut self, output: usize) {
        if self.tensors[output].requires_grad {
            let len = self.tensors[output].data.len();
            self.tensors[output].grad = vec![1.0; len];
        } else {
            panic!("Cannot call backward on a tensor doesn't require gradients");
        }

        for i in (0..=output).rev() {
            let (op, grad_current, _data_current) = {
                let t = &self.tensors[i];
                (t.op.clone(), t.grad.clone(), t.data.clone())
            };

            match op {
                Op::None => {}

                Op::Add(a, b) => {
                    if let Some(parent) = self.tensors.get_mut(a) {
                        if parent.requires_grad {
                            for (g, pg) in grad_current.iter().zip(parent.grad.iter_mut()) {
                                *pg += *g;
                            }
                        }
                    }

                    if let Some(parent) = self.tensors.get_mut(b) {
                        if parent.requires_grad {
                            for (g, pg) in grad_current.iter().zip(parent.grad.iter_mut()) {
                                *pg += *g;
                            }
                        }
                    }
                }

                Op::Mul(a, b) => {
                    let data_a = self.tensors[a].data.clone();
                    let data_b = self.tensors[b].data.clone();

                    if let Some(parent) = self.tensors.get_mut(a) {
                        if parent.requires_grad {
                            for ((g, d), pg) in grad_current
                                .iter()
                                .zip(data_b.iter())
                                .zip(parent.grad.iter_mut())
                            {
                                *pg += *g * *d;
                            }
                        }
                    }

                    if let Some(parent) = self.tensors.get_mut(b) {
                        if parent.requires_grad {
                            for ((g, d), pg) in grad_current
                                .iter()
                                .zip(data_a.iter())
                                .zip(parent.grad.iter_mut())
                            {
                                *pg += *g * *d;
                            }
                        }
                    }
                }

                Op::MatMul(a, b) => {
                    let data_a = self.tensors[a].data.clone();
                    let data_b = self.tensors[b].data.clone();
                    let shape_a = self.tensors[a].shape.clone();
                    let shape_b = self.tensors[b].shape.clone();
                    let req_a = self.tensors[a].requires_grad;
                    let req_b = self.tensors[b].requires_grad;

                    let m = shape_a[0];
                    let k = shape_a[1];
                    let n = shape_b[1];

                    if req_a {
                        let b_t = raw_transpose(&data_b, k, n);
                        let grad_a_contrib = raw_matmul(&grad_current, &b_t, m, n, k);
                        for (g, pg) in grad_a_contrib.iter().zip(self.tensors[a].grad.iter_mut()) {
                            *pg += *g;
                        }
                    }

                    if req_b {
                        let a_t = raw_transpose(&data_a, m, k);
                        let grad_b_contrib = raw_matmul(&a_t, &grad_current, k, m, n);
                        for (g, pg) in grad_b_contrib.iter().zip(self.tensors[b].grad.iter_mut()) {
                            *pg += *g;
                        }
                    }
                }

                Op::AddBroadcast(mat_idx, bias_idx) => {
                    let mat_shape = self.tensors[mat_idx].shape.clone();
                    let mat_req = self.tensors[mat_idx].requires_grad;
                    let bias_req = self.tensors[bias_idx].requires_grad;

                    let m = mat_shape[0];
                    let n = mat_shape[1];

                    if mat_req {
                        for (g, mg) in grad_current
                            .iter()
                            .zip(self.tensors[mat_idx].grad.iter_mut())
                        {
                            *mg += *g;
                        }
                    }

                    if bias_req {
                        let mut bias_grad = vec![0.0; n];
                        for row in 0..m {
                            for col in 0..n {
                                bias_grad[col] += grad_current[row * n + col];
                            }
                        }

                        for (bg, bg_mut) in
                            bias_grad.iter().zip(self.tensors[bias_idx].grad.iter_mut())
                        {
                            *bg_mut += *bg;
                        }
                    }
                }

                Op::ReLU(parent_idx) => {
                    let parent_data = self.tensors[parent_idx].data.clone();
                    let parent_req = self.tensors[parent_idx].requires_grad;

                    if parent_req {
                        for (i, (g, &p_val)) in
                            grad_current.iter().zip(parent_data.iter()).enumerate()
                        {
                            if p_val > 0.0 {
                                self.tensors[parent_idx].grad[i] += *g;
                            }
                        }
                    }
                }

                Op::MSE(pred_idx, target_idx) => {
                    let pred_data = self.tensors[pred_idx].data.clone();
                    let target_data = self.tensors[target_idx].data.clone();
                    let n = pred_data.len() as f64;
                    let upstream_grad = grad_current[0];

                    if self.tensors[pred_idx].requires_grad {
                        for (i, (p, t)) in pred_data.iter().zip(target_data.iter()).enumerate() {
                            let local_grad = (2.0 / n) * (p - t);
                            self.tensors[pred_idx].grad[i] += upstream_grad * local_grad;
                        }
                    }
                }

                Op::Conv2D {
                    input,
                    kernel,
                    bias,
                    in_channels,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    in_h,
                    in_w,
                } => {
                    let out_h = in_h - kernel_h + 1;
                    let out_w = in_w - kernel_w + 1;

                    let input_data = self.tensors[input].data.clone();
                    let kernel_data = self.tensors[kernel].data.clone();
                    let req_input = self.tensors[input].requires_grad;
                    let req_kernel = self.tensors[kernel].requires_grad;
                    let req_bias = self.tensors[bias].requires_grad;

                    if req_kernel {
                        for oc in 0..out_channels {
                            for r in 0..out_h {
                                for c in 0..out_w {
                                    let grad_val = grad_current[oc * out_h * out_w + r * out_w + c];
                                    for ic in 0..in_channels {
                                        for kr in 0..kernel_h {
                                            for kc in 0..kernel_w {
                                                let input_idx_flat =
                                                    ic * in_h * in_w + (r + kr) * in_w + (c + kc);
                                                let kernel_idx_flat =
                                                    oc * in_channels * kernel_h * kernel_w
                                                        + ic * kernel_h * kernel_w
                                                        + kr * kernel_w
                                                        + kc;
                                                self.tensors[kernel].grad[kernel_idx_flat] +=
                                                    grad_val * input_data[input_idx_flat];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if req_input {
                        for oc in 0..out_channels {
                            for r in 0..out_h {
                                for c in 0..out_w {
                                    let grad_val = grad_current[oc * out_h * out_w + r * out_w + c];
                                    for ic in 0..in_channels {
                                        for kr in 0..kernel_h {
                                            for kc in 0..kernel_w {
                                                let input_idx_flat =
                                                    ic * in_h * in_w + (r + kr) * in_w + (c + kc);
                                                let kernel_idx_flat =
                                                    oc * in_channels * kernel_h * kernel_w
                                                        + ic * kernel_h * kernel_w
                                                        + kr * kernel_w
                                                        + kc;
                                                self.tensors[input].grad[input_idx_flat] +=
                                                    grad_val * kernel_data[kernel_idx_flat];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if req_bias {
                        for oc in 0..out_channels {
                            let mut sum = 0.0;
                            for r in 0..out_h {
                                for c in 0..out_w {
                                    sum += grad_current[oc * out_h * out_w + r * out_w + c];
                                }
                            }
                            self.tensors[bias].grad[oc] += sum;
                        }
                    }
                }

                Op::MaxPool2D {
                    input,
                    channels,
                    in_h,
                    in_w,
                    pool_size,
                } => {
                    let input_data = self.tensors[input].data.clone();
                    let req_input = self.tensors[input].requires_grad;
                    let out_h = in_h / pool_size;
                    let out_w = in_w / pool_size;

                    if req_input {
                        for ch in 0..channels {
                            for r in 0..out_h {
                                for c in 0..out_w {
                                    // find which posisitional had the max
                                    let mut max_val = f64::NEG_INFINITY;
                                    let mut max_idx = 0;
                                    for pr in 0..pool_size {
                                        for pc in 0..pool_size {
                                            let in_r = r * pool_size + pr;
                                            let in_c = c * pool_size + pc;
                                            let idx = ch * in_h * in_w + in_r * in_w + in_c;
                                            if input_data[idx] > max_val {
                                                max_val = input_data[idx];
                                                max_idx = idx;
                                            }
                                        }
                                    }
                                    // only the max posisitons gets the gradients
                                    let grad_val = grad_current[ch * out_h * out_w + r * out_w + c];
                                    self.tensors[input].grad[max_idx] += grad_val;
                                }
                            }
                        }
                    }
                }

                Op::Flatten {
                    input,
                    channels,
                    h,
                    w,
                } => {
                    if self.tensors[input].requires_grad {
                        for (i, g) in grad_current.iter().enumerate() {
                            self.tensors[input].grad[i] += *g;
                        }
                    }
                }
                Op::Softmax { input, cols } => {
                    let total_len = _data_current.len();
                    let rows = total_len / cols;

                    if self.tensors[input].requires_grad {
                        let mut input_grad = vec![0.0; total_len];

                        for r in 0..rows {
                            let start = r * cols;
                            let end = start + cols;

                            let s_row = &_data_current[start..end];
                            let grad_row = &grad_current[start..end];

                            let mut dot = 0.0;
                            for j in 0..cols {
                                dot += grad_row[j] * s_row[j];
                            }

                            for j in 0..cols {
                                input_grad[start + j] = s_row[j] * (grad_row[j] - dot);
                            }
                        }

                        for (idx, g) in input_grad.into_iter().enumerate() {
                            self.tensors[input].grad[idx] += g;
                        }
                    }
                }

                Op::ScalarMul { input, scalar } => {
                    if self.tensors[input].requires_grad {
                        for (i, g) in grad_current.iter().enumerate() {
                            self.tensors[input].grad[i] += g * scalar;
                        }
                    }
                }

                Op::Reshape { input } => {
                    if self.tensors[input].requires_grad {
                        for (i, g) in grad_current.iter().enumerate() {
                            self.tensors[input].grad[i] += g;
                        }
                    }
                }

                Op::Transpose { input, dim0, dim1 } => {
                    if self.tensors[input].requires_grad {
                        let old_shape = self.tensors[input].shape.clone();
                        let ndim = old_shape.len();

                        // current tensor already swap
                        // need to untranspose it the gradient back
                        let mut swapped_shape = old_shape.clone();
                        swapped_shape.swap(dim0, dim1);

                        let total_len = grad_current.len();

                        for swapped_flat in 0..total_len {
                            // conver flat to coords using swapped_shape
                            let mut coords = vec![0usize; ndim];
                            let mut remaining = swapped_flat;
                            for d in 0..ndim {
                                let stride: usize = swapped_shape[d + 1..].iter().product();
                                coords[d] = remaining / stride;
                                remaining %= stride;
                            }

                            // swap back
                            coords.swap(dim0, dim1);
                            let mut orig_flat = 0;
                            for d in 0..ndim {
                                let stride: usize = old_shape[d + 1..].iter().product();
                                orig_flat += coords[d] * stride;
                            }

                            self.tensors[input].grad[orig_flat] += grad_current[swapped_flat];
                        }
                    }
                }

                Op::SoftmaxCrossEntropy {
                    input,
                    target,
                    cols,
                } => {
                    if self.tensors[input].requires_grad {
                        // softmax output is stored at index [i - 1]
                        // because pushed it right before the loss tensor
                        let sm_data = self.tensors[i - 1].data.clone();
                        let target_data = self.tensors[target].data.clone();
                        let rows = sm_data.len() / cols;
                        let upstream = grad_current[0];

                        for j in 0..sm_data.len() {
                            self.tensors[input].grad[j] +=
                                upstream * (sm_data[j] - target_data[j]) / rows as f64;
                        }
                    }
                }

                Op::LayerNorm {
                    input,
                    gamma,
                    beta,
                    cols,
                    eps,
                } => {
                    let input_data = self.tensors[input].data.clone();
                    let gamma_data = self.tensors[gamma].data.clone();
                    let norm_data = self.tensors[i - 1].data.clone(); // hidden tensor
                    let req_input = self.tensors[input].requires_grad;
                    let req_gamma = self.tensors[gamma].requires_grad;
                    let req_beta = self.tensors[beta].requires_grad;
                    let total_len = input_data.len();
                    let rows = total_len / cols;

                    // dL/d_gamma
                    if req_gamma {
                        for r in 0..rows {
                            let start = r * cols;
                            for c in 0..cols {
                                self.tensors[gamma].grad[c] +=
                                    grad_current[start + c] * norm_data[start + c];
                            }
                        }
                    }

                    // dL/d_beta
                    if req_beta {
                        for r in 0..rows {
                            let start = r * cols;
                            for c in 0..cols {
                                self.tensors[beta].grad[c] += grad_current[start + c];
                            }
                        }
                    }

                    // dL/d_x
                    if req_input {
                        for r in 0..rows {
                            let start = r * cols;
                            // compute mean and variance again :)
                            let mut mean = 0.0;
                            for c in 0..cols {
                                mean += input_data[start + c];
                            }
                            mean /= cols as f64;

                            let mut var = 0.0;
                            for c in 0..cols {
                                let diff = input_data[start + c] - mean;
                                var += diff * diff;
                            }
                            var /= cols as f64;

                            let inv_std = 1.0 / (var + eps).sqrt();

                            // Hard compute stuff
                            let mut mean_dy_dnorm = 0.0;
                            let mut mean_dy_dnorm_norm = 0.0;
                            for c in 0..cols {
                                let dy_dnorm = grad_current[start + c] * gamma_data[c];
                                mean_dy_dnorm += dy_dnorm;
                                mean_dy_dnorm_norm += dy_dnorm * norm_data[start + c];
                            }

                            mean_dy_dnorm /= cols as f64;
                            mean_dy_dnorm_norm /= cols as f64;

                            // final gradient input
                            for c in 0..cols {
                                let dy_dnorm = grad_current[start + c] * gamma_data[c];
                                self.tensors[input].grad[start + c] += inv_std
                                    * (dy_dnorm
                                        - mean_dy_dnorm
                                        - norm_data[start + c] * mean_dy_dnorm_norm);
                            }
                        }
                    }
                }

                Op::Embedding { table, ref indices } => {
                    if self.tensors[table].requires_grad {
                        let d_model = self.tensors[table].shape[1];
                        for (i, &idx) in indices.iter().enumerate() {
                            let grad_start = i * d_model;
                            let table_start = idx * d_model;
                            for d in 0..d_model {
                                self.tensors[table].grad[table_start + d] +=
                                    grad_current[grad_start + d];
                            }
                        }
                    }
                }

                Op::SliceCols {
                    input,
                    rows,
                    total_cols,
                    col_start,
                    col_end,
                } => {
                    if self.tensors[input].requires_grad {
                        let slice_cols = col_end - col_start;
                        for r in 0..rows {
                            for c in 0..slice_cols {
                                self.tensors[input].grad[r * total_cols + col_start + c] +=
                                    grad_current[r * slice_cols + c];
                            }
                        }
                    }
                }

                Op::ConcatCols { ref inputs, rows } => {
                    let total_cols: usize =
                        inputs.iter().map(|&idx| self.tensors[idx].shape[1]).sum();
                    let mut col_offset = 0;
                    for &idx in inputs.iter() {
                        let cols = self.tensors[idx].shape[1];
                        if self.tensors[idx].requires_grad {
                            for r in 0..rows {
                                for c in 0..cols {
                                    self.tensors[idx].grad[r * cols + c] +=
                                        grad_current[r * total_cols + col_offset + c];
                                }
                            }
                        }
                        col_offset += cols;
                    }
                }

                Op::MeanPool { input, rows, cols } => {
                    if self.tensors[input].requires_grad {
                        for r in 0..rows {
                            for c in 0..cols {
                                self.tensors[input].grad[r * cols + c] +=
                                    grad_current[c] / rows as f64;
                            }
                        }
                    }
                }
            }
        }
    }
}
