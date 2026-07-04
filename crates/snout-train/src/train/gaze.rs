use std::sync::Arc;

use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;

use crate::data::Frame;
use crate::data::batch::SampleBatcher;
use crate::data::stream::GazeStream;
use crate::model::{EyeNet, pretrained_gaze};
use crate::train::config::TrainConfig;
use crate::train::progress::Reporter;
use crate::train::schedule::WarmupCosine;

/// Trains the gaze net ([`EyeNet`]) on the stratified gaze stream.
pub struct GazeTrainer<'a, B: AutodiffBackend> {
    frames: Arc<Vec<Frame>>,
    config: &'a TrainConfig,
    device: &'a B::Device,
    reporter: Reporter<'a>,
}

impl<'a, B: AutodiffBackend> GazeTrainer<'a, B> {
    pub fn new(
        frames: Arc<Vec<Frame>>,
        config: &'a TrainConfig,
        device: &'a B::Device,
        reporter: Reporter<'a>,
    ) -> Self {
        Self { frames, config, device, reporter }
    }

    pub fn fit(mut self) -> EyeNet<B> {
        let total = self.config.gaze_steps;

        // One `.iter()` yields exactly `total` batches (SamplerDataset-pinned), so the
        // loop never re-iterates.
        let dataset = GazeStream::new(&self.frames, total * self.config.batch_size).build();

        // Loader is single-threaded to avoid memory pool fragmentation from dropped worker threads.
        // See https://github.com/tracel-ai/burn/issues/4991.
        let loader = DataLoaderBuilder::new(SampleBatcher)
            .batch_size(self.config.batch_size)
            .shuffle(self.config.seed)
            .set_device(self.device.clone())
            .build(dataset);

        let mut model = pretrained_gaze::<B>(self.device);
        let mut optim = AdamWConfig::new().init();
        let schedule = WarmupCosine::gaze(total);

        let mut step = 0;
        while step < total {
            for batch in loader.iter() {
                if step >= total {
                    break;
                }

                let lr = schedule.lr_at(step);
                let loss = MseLoss::new().forward(model.forward(batch.inputs), batch.gaze, Reduction::Mean);
                self.reporter.report(&loss, step, total, None);
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(lr, model, grads);
                step += 1;
            }
        }

        model
    }
}
