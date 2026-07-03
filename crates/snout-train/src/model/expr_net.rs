use burn::tensor::backend::Backend;
use mobilenetv4_burn::model::{MobileNetV4, MobileNetV4Config};

use crate::spec::{EXPR_OUTPUTS, PER_EYE_CHANNELS};

/// The expression tower: MobileNetV4 `conv_small` x0.5, producing four raw linear
/// outputs (`[lid, widen, squint, brow]`) from a per-eye temporal stack.
pub type ExprNet<B> = MobileNetV4<B>;

const DROPOUT: f64 = 0.2;

/// Builds the (untrained) expression net configured for the per-eye temporal stack.
pub fn expr_net<B: Backend>(device: &B::Device) -> ExprNet<B> {
    MobileNetV4Config::new()
        .with_in_channels(PER_EYE_CHANNELS)
        .with_num_classes(EXPR_OUTPUTS)
        .with_dropout(DROPOUT)
        .init(device)
}
