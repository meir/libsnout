//! Held-out validation for the expression net.

use std::sync::Arc;

use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;

use crate::data::batch::{SampleBatch, SampleBatcher};
use crate::data::dataset::SampleItem;
use crate::model::ExprNet;
use crate::spec::EXPR_OUTPUTS;
use crate::train::progress::ExprMetrics;

/// Runs held-out validation of the expression net on a fixed step cadence.
///
/// Wraps the held-out dataset in an inner-backend [`DataLoader`] and decides when a step
/// is due, so the training loop only asks [`Validator::at`] and routes any returned
/// metrics to its reporter -- keeping the "what/when to evaluate" concern out of the
/// trainer.
pub struct Validator<B: AutodiffBackend> {
    loader: Arc<dyn DataLoader<B::InnerBackend, SampleBatch<B::InnerBackend>>>,
    every: usize,
    empty: bool,
}

impl<B: AutodiffBackend> Validator<B> {
    /// Builds a loader over the held-out `dataset`. Single-threaded and unshuffled:
    /// validation runs repeatedly, and a dropped worker's device pool isn't reclaimed on
    /// wgpu/CubeCL (burn #4792/#4991); one clean pass scores each sample once.
    pub fn new<D>(dataset: D, device: &B::Device, batch_size: usize, every: usize) -> Self
    where
        D: Dataset<SampleItem> + Send + Sync + 'static,
    {
        let empty = dataset.is_empty();
        let loader = DataLoaderBuilder::new(SampleBatcher)
            .batch_size(batch_size.max(1))
            .set_device(device.clone())
            .build(dataset);
        Self { loader, every, empty }
    }

    /// Metrics when `step` lands on the cadence (or is the final step) and there is
    /// held-out data; otherwise `None`.
    pub fn at(&self, model: &ExprNet<B>, step: usize, total: usize) -> Option<ExprMetrics> {
        let due = self.every > 0 && !self.empty && (step % self.every == 0 || step + 1 == total);
        due.then(|| self.evaluate(model))
    }

    /// One pass over the held-out loader on the inner (inference) backend, scoring mean
    /// squared error over the four expression channels plus per-channel Pearson
    /// correlation. Predictions are clamped to `[0, 1]` (matching the reference).
    fn evaluate(&self, model: &ExprNet<B>) -> ExprMetrics {
        let eval = model.valid();

        // Streaming accumulators for MSE and per-channel Pearson correlation.
        let mut sq_err = 0.0f64;
        let mut n = 0u64;
        let mut sum_p = [0.0f64; EXPR_OUTPUTS];
        let mut sum_t = [0.0f64; EXPR_OUTPUTS];
        let mut sum_pp = [0.0f64; EXPR_OUTPUTS];
        let mut sum_tt = [0.0f64; EXPR_OUTPUTS];
        let mut sum_pt = [0.0f64; EXPR_OUTPUTS];

        for batch in self.loader.iter() {
            let preds = eval.forward(batch.inputs).clamp(0.0, 1.0);
            let pred: Vec<f32> = preds.into_data().to_vec().expect("f32 predictions");
            let tgt: Vec<f32> = batch.expr.into_data().to_vec().expect("f32 targets");

            for row in 0..tgt.len() / EXPR_OUTPUTS {
                for c in 0..EXPR_OUTPUTS {
                    let p = pred[row * EXPR_OUTPUTS + c] as f64;
                    let t = tgt[row * EXPR_OUTPUTS + c] as f64;
                    let d = p - t;
                    sq_err += d * d;
                    sum_p[c] += p;
                    sum_t[c] += t;
                    sum_pp[c] += p * p;
                    sum_tt[c] += t * t;
                    sum_pt[c] += p * t;
                }
                n += 1;
            }
        }

        let nf = n.max(1) as f64;
        let mse = (sq_err / (nf * EXPR_OUTPUTS as f64)) as f32;
        let mut corr = [0.0f32; EXPR_OUTPUTS];
        for c in 0..EXPR_OUTPUTS {
            let cov = nf * sum_pt[c] - sum_p[c] * sum_t[c];
            let var_p = nf * sum_pp[c] - sum_p[c] * sum_p[c];
            let var_t = nf * sum_tt[c] - sum_t[c] * sum_t[c];
            corr[c] = if var_p > 1e-9 && var_t > 1e-9 {
                (cov / (var_p.sqrt() * var_t.sqrt())) as f32
            } else {
                0.0
            };
        }

        ExprMetrics { mse, corr }
    }
}
