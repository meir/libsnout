use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::model::eye_net::EyeNet;
use crate::model::expr_net::{ExprNet, expr_net};

const GAZE_PRETRAIN: &[u8] = include_bytes!("../../assets/gaze_pretrain.safetensors");
const EXPR_PRETRAIN: &[u8] = include_bytes!("../../assets/expr_pretrain.safetensors");

/// Loads the pretrained gaze net (MicroChad warm-start).
pub fn pretrained_gaze<B: Backend>(device: &B::Device) -> EyeNet<B> {
    let mut model = EyeNet::new(device);
    load(&mut model, GAZE_PRETRAIN, "gaze");
    model
}

/// Loads the pretrained expression net (MobileNetV4 warm-start).
pub fn pretrained_expr<B: Backend>(device: &B::Device) -> ExprNet<B> {
    let mut model = expr_net(device);
    load(&mut model, EXPR_PRETRAIN, "expr");
    model
}

/// Loads `bytes` into `model`, asserting a complete (no missing / no error) load.
fn load<B: Backend, M: ModuleSnapshot<B>>(model: &mut M, bytes: &[u8], name: &str) {
    let mut store = SafetensorsStore::from_bytes(Some(bytes.to_vec()));
    let result = model
        .load_from(&mut store)
        .unwrap_or_else(|e| panic!("load {name} pretrain: {e:?}"));
    assert!(result.errors.is_empty(), "{name} pretrain: {:?}", result.errors);
    assert!(result.missing.is_empty(), "{name} pretrain missing: {:?}", result.missing);
}
