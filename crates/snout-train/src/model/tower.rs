use burn::prelude::*;

use crate::model::eye_net::EyeNet;
use crate::model::expr_net::{ExprNet, expr_net};

/// Dual-task tower that processes a single eye's temporal stack through two heads:
/// - **Gaze**: [`EyeNet`] (MicroChad) -> 2 outputs with sigmoid,
/// - **Expression**: [`ExprNet`] (MobileNetV4) -> 4 raw linear outputs.
///
/// Forward produces `[B, 6]` = `cat([gaze, expr])`.
#[derive(Module, Debug)]
pub struct DualTaskTower<B: Backend> {
    pub gaze: EyeNet<B>,
    pub expr: ExprNet<B>,
}

impl<B: Backend> DualTaskTower<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            gaze: EyeNet::new(device),
            expr: expr_net(device),
        }
    }

    /// `[B, 4, 128, 128] -> [B, 6]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let gaze = self.gaze.forward(x.clone());
        let expr = self.expr.forward(x);
        Tensor::cat(vec![gaze, expr], 1)
    }
}
