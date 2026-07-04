//! ONNX export by swapping trained weights into a precomputed template.
//!
//! burn cannot emit ONNX, so we ship a structural template (`merged_template.onnx`,
//! generated once by `tools/export_template.py`) and replace its initializers with
//! the trained weights. The template is a fused, conv-only graph: each burn
//! `conv + BatchNorm` pair is folded into a single biased conv (standard inference
//! folding), and linear weights are transposed to PyTorch `[out, in]` layout.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use onnx_rs::ast::TensorProto;

use crate::model::MergedDualEye;

const TEMPLATE: &[u8] = include_bytes!("../assets/merged_template.onnx");
const BN_EPSILON: f32 = 1e-5;

/// Exports the trained model to an ONNX file at `path`.
pub fn export_onnx<B: Backend>(
    model: &MergedDualEye<B>,
    path: impl AsRef<Path>,
) -> Result<(), String> {
    let weights = onnx_initializers(model);

    // Bind the template to a local so the parsed model borrows a non-'static
    // lifetime, letting us splice in locally-owned replacement bytes.
    let template: &[u8] = TEMPLATE;
    let mut onnx = onnx_rs::parse(template).map_err(|e| format!("parse template: {e:?}"))?;

    let mut replaced = 0usize;
    let mut total = 0usize;
    if let Some(graph) = onnx.graph.as_mut() {
        total = graph.initializer.len();
        for init in graph.initializer.iter_mut() {
            if let Some(bytes) = weights.get(init.name()) {
                *init = TensorProto::from_raw(
                    init.name(),
                    init.dims().to_vec(),
                    init.data_type(),
                    bytes,
                );
                replaced += 1;
            }
        }
    }

    // Every produced weight must land on a template initializer...
    if replaced != weights.len() {
        return Err(format!(
            "produced {} weights but only {replaced} matched a template initializer (name mismatch)",
            weights.len()
        ));
    }
    // ...and every template initializer must be overwritten, or it would keep the
    // template's (stripped/placeholder) values in the exported model.
    if replaced != total {
        return Err(format!(
            "{} of {total} template initializers were not overwritten",
            total - replaced
        ));
    }

    let out = onnx_rs::encode(&onnx);
    std::fs::write(path, out).map_err(|e| format!("write onnx: {e}"))
}

/// Collects the model's tensors and produces the ONNX initializer name -> little-endian
/// f32 byte payload map (conv+BN folded, linears transposed).
fn onnx_initializers<B: Backend>(model: &MergedDualEye<B>) -> HashMap<String, Vec<u8>> {
    // name -> (values, shape)
    let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for snap in model.collect(None, None, true) {
        let data = snap.to_data().expect("materialize tensor");
        let shape = data.shape.to_vec();
        let values = data.to_vec::<f32>().expect("f32 tensor");
        raw.insert(snap.full_path(), (values, shape));
    }

    let mut out: HashMap<String, Vec<u8>> = HashMap::new();
    let mut consumed: HashSet<String> = HashSet::new();

    // Fold each conv+BN (a burn `ConvNorm`) into a single biased conv.
    for name in raw.keys().cloned().collect::<Vec<_>>() {
        let Some(prefix) = name.strip_suffix(".bn.gamma") else {
            continue;
        };
        let (w, w_shape) = &raw[&format!("{prefix}.conv.weight")];
        let gamma = &raw[&format!("{prefix}.bn.gamma")].0;
        let beta = &raw[&format!("{prefix}.bn.beta")].0;
        let mean = &raw[&format!("{prefix}.bn.running_mean")].0;
        let var = &raw[&format!("{prefix}.bn.running_var")].0;

        let (folded_w, folded_b) = fold_conv_bn(w, w_shape[0], gamma, beta, mean, var);
        out.insert(format!("{prefix}.conv.weight"), to_le_bytes(&folded_w));
        out.insert(format!("{prefix}.conv.bias"), to_le_bytes(&folded_b));

        for suffix in ["conv.weight", "bn.gamma", "bn.beta", "bn.running_mean", "bn.running_var"] {
            consumed.insert(format!("{prefix}.{suffix}"));
        }
    }

    // Remaining tensors: gaze convs (copy), linear weights (transpose), biases (copy).
    for (name, (values, shape)) in &raw {
        if consumed.contains(name) || name.contains(".bn.") {
            continue;
        }
        if shape.len() == 2 {
            // Linear weight: burn [in, out] -> ONNX/PyTorch [out, in].
            out.insert(name.clone(), to_le_bytes(&transpose(values, shape[0], shape[1])));
        } else {
            out.insert(name.clone(), to_le_bytes(values));
        }
    }

    out
}

/// Folds `conv (no bias) + BatchNorm` into an equivalent biased conv.
/// `scale = gamma / sqrt(var + eps)`, `W' = W * scale`, `b' = beta - mean * scale`.
fn fold_conv_bn(
    w: &[f32],
    out_channels: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let per_channel = w.len() / out_channels;
    let mut folded_w = vec![0.0; w.len()];
    let mut folded_b = vec![0.0; out_channels];

    for o in 0..out_channels {
        let scale = gamma[o] / (var[o] + BN_EPSILON).sqrt();
        for i in 0..per_channel {
            folded_w[o * per_channel + i] = w[o * per_channel + i] * scale;
        }
        folded_b[o] = beta[o] - mean[o] * scale;
    }

    (folded_w, folded_b)
}

/// Transposes a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

fn to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};

    use crate::model::DualTaskTower;

    type B = NdArray;

    #[test]
    fn export_roundtrip_dumps_reference() {
        let device = Default::default();
        let merged = MergedDualEye::new(DualTaskTower::new(&device));

        // Deterministic input both Rust and onnxruntime can reproduce exactly.
        let n = 8 * 128 * 128;
        let input: Vec<f32> = (0..n).map(|i| (i % 256) as f32 / 255.0).collect();
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(input.clone(), [1, 8, 128, 128]),
            &device,
        );
        let out = merged.forward(x).to_data().to_vec::<f32>().unwrap();

        export_onnx(&merged, "/tmp/test_merged.onnx").expect("export");
        std::fs::write("/tmp/merged_input.bin", to_le_bytes(&input)).unwrap();
        std::fs::write("/tmp/merged_burn_out.bin", to_le_bytes(&out)).unwrap();
        println!("burn output: {out:?}");
    }
}
