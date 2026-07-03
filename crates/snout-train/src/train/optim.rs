//! The split optimizer: updates the classifier head and backbone at independent rates.

use burn::module::ParamId;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

use crate::model::ExprNet;

/// An optimizer wrapper that updates the expression net's classifier head and its
/// backbone at independent learning rates. A backbone rate of `0.0` freezes the
/// backbone for that step, leaving only the head to update.
///
/// Mirrors the two parameter groups in the Python expression trainer.
pub struct SplitOptimizer<O> {
    optim: O,
    head_ids: Vec<ParamId>,
}

impl<O> SplitOptimizer<O> {
    /// Backpropagates `loss`, then steps the head and (unless frozen) the backbone.
    pub fn step<B>(
        &mut self,
        head_lr: f64,
        backbone_lr: f64,
        mut model: ExprNet<B>,
        loss: Tensor<B, 1>,
    ) -> ExprNet<B>
    where
        B: AutodiffBackend,
        O: Optimizer<ExprNet<B>, B>,
    {
        // `from_params` removes the head gradients, leaving the backbone in `raw`.
        let mut raw = loss.backward();
        let head = GradientsParams::from_params(&mut raw, &model, &self.head_ids);
        model = self.optim.step(head_lr, model, head);

        if backbone_lr > 0.0 {
            let backbone = GradientsParams::from_grads(raw, &model);
            model = self.optim.step(backbone_lr, model, backbone);
        }

        model
    }
}

/// Builds an AdamW split optimizer that partitions model's classifier head from its backbone.
pub fn adamw_split<B: AutodiffBackend>(
    model: &ExprNet<B>,
) -> SplitOptimizer<impl Optimizer<ExprNet<B>, B> + use<B>> {
    let mut head_ids = vec![model.classifier.weight.id];
    if let Some(bias) = &model.classifier.bias {
        head_ids.push(bias.id);
    }

    SplitOptimizer {
        optim: AdamWConfig::new().init::<B, ExprNet<B>>(),
        head_ids,
    }
}
