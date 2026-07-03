use std::sync::Arc;

use burn::data::dataloader::DataLoaderBuilder;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

use crate::data::Frame;
use crate::data::batch::{PairedSampleBatcher, SampleBatcher};
use crate::data::stream::{EXPR_EVEN_BATCH, ExprData};
use crate::model::{ExprNet, pretrained_expr};
use crate::train::config::TrainConfig;
use crate::train::eval::Validator;
use crate::train::optim::adamw_split;
use crate::train::progress::Reporter;
use crate::train::schedule::ExprSchedule;
use crate::train::loss::{self, DeconfuseLoss, EvennessLoss};

/// Trains the expression net ([`ExprNet`]). A single AdamW optimizer runs the whole
/// schedule ([`ExprSchedule`]): the head warms up with the backbone frozen, both
/// train at the phase-A rates, then drop to the lower phase-B rates at the midpoint
/// (Python EXPR_STAGE_A/B), with momentum carried across the boundary.
pub struct ExprTrainer<'a, B: AutodiffBackend> {
    frames: Arc<Vec<Frame>>,
    config: &'a TrainConfig,
    device: &'a B::Device,
    reporter: Reporter<'a>,
    deconfuse: DeconfuseLoss<B>,
    evenness: EvennessLoss<B>,
}

impl<'a, B: AutodiffBackend> ExprTrainer<'a, B> {
    pub fn new(
        frames: Arc<Vec<Frame>>,
        config: &'a TrainConfig,
        device: &'a B::Device,
        reporter: Reporter<'a>,
    ) -> Self {
        Self {
            frames,
            config,
            device,
            reporter,
            deconfuse: DeconfuseLoss::new(device),
            evenness: EvennessLoss::new(device),
        }
    }

    pub fn fit(mut self) -> ExprNet<B> {
        let total = self.config.expr_steps;
        let schedule = ExprSchedule::new(total);

        let data = ExprData::new(&self.frames, self.config);

        let validator = Validator::<B>::new(
            data.c.build(),
            self.device,
            self.config.batch_size,
            self.config.val_every,
        );

        // One `.iter()` yields exactly `total` batches (streams are `SamplerDataset`-
        // pinned), so the loop never re-iterates -- which would respawn, and on
        // wgpu/CubeCL strand, the worker pool (burn #4792/#4991).
        let dataset = data.a.build();
        let loader = DataLoaderBuilder::new(SampleBatcher)
            .batch_size(self.config.batch_size)
            .shuffle(self.config.seed)
            .num_workers(self.config.num_workers)
            .set_device(self.device.clone())
            .build(dataset);

        // Paired-eye loader for the evenness term (skipped when there are no L/R pairs),
        // likewise pinned to one batch per step so `paired_iter` never runs dry.
        let paired_loader = data.b
            .build()
            .map(|paired| {
                DataLoaderBuilder::new(PairedSampleBatcher)
                    .batch_size(EXPR_EVEN_BATCH)
                    .shuffle(self.config.seed)
                    .num_workers(self.config.num_workers.min(2))
                    .set_device(self.device.clone())
                    .build(paired)
            });

        let mut model = pretrained_expr::<B>(self.device);
        let mut optim = adamw_split(&model);
        let mut paired_iter = paired_loader.as_ref().map(|pl| pl.iter());

        let mut step = 0;
        while step < total {
            for batch in loader.iter() {
                if step >= total {
                    break;
                }
                let (head_lr, backbone_lr) = schedule.lrs_at(step);

                // One forward over [main; paired-left; paired-right] so BatchNorm sees
                // the combined batch (matches the Python `net(torch.cat(parts))`).
                let main_n = batch.inputs.dims()[0];
                let (pred, even) = match paired_iter.as_mut() {
                    Some(iter) => {
                        let pair = iter.next().expect("paired loader outlasts the step count");
                        let pair_n = pair.left_inputs.dims()[0];
                        let outs = model.forward(Tensor::cat(
                            vec![batch.inputs, pair.left_inputs, pair.right_inputs],
                            0,
                        ));
                        let pred = outs.clone().narrow(0, 0, main_n);
                        let pred_l = outs.clone().narrow(0, main_n, pair_n);
                        let pred_r = outs.narrow(0, main_n + pair_n, pair_n);

                        let even = self.evenness.forward(pred_l, pred_r, pair.left_expr, pair.right_expr);

                        (pred, Some(even))
                    }
                    None => (model.forward(batch.inputs), None),
                };

                let reg = self.deconfuse.forward(pred.clone(), batch.expr);
                let range = loss::range_hinge(pred).mul_scalar(loss::LAMBDA_RANGE);
                let mut loss = reg + range;
                if let Some(even) = even {
                    loss = loss + even;
                }

                let metrics = validator.at(&model, step, total);
                self.reporter.report(&loss, step, total, metrics);
                model = optim.step(head_lr, backbone_lr, model, loss);
                step += 1;
            }
        }

        model
    }
}
