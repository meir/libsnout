//! Training: the orchestrator plus the per-net trainers and their supporting types.

mod config;
mod eval;
mod expr;
mod gaze;
mod loss;
mod optim;
mod progress;
mod schedule;

use burn::module::AutodiffModule;
pub use config::TrainConfig;
pub use eval::Validator;
pub use expr::ExprTrainer;
pub use gaze::GazeTrainer;
pub use progress::{ExprMetrics, Phase, Progress, Reporter, TrainError};

use std::path::Path;
use std::sync::Arc;

use burn::tensor::backend::AutodiffBackend;

use crate::data;
use crate::model::{DualTaskTower, MergedDualEye};

/// Fluent trainer: configure, attach a progress callback, then run.
///
/// ```no_run
/// # use snout_train::train::{Trainer, TrainConfig};
/// # fn run<B: burn::tensor::backend::AutodiffBackend>(device: B::Device) {
/// Trainer::<B>::new(device)
///     .config(TrainConfig::default())
///     .on_progress(|p| println!("[{}] {}/{} loss={:.4}", p.phase, p.step, p.total_steps, p.loss))
///     .train_to_onnx("capture.bin", "model.onnx")
///     .unwrap();
/// # }
/// ```
pub struct Trainer<B: AutodiffBackend> {
    config: TrainConfig,
    device: B::Device,
    on_progress: Option<Box<dyn FnMut(Progress)>>,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            config: TrainConfig::default(),
            device,
            on_progress: None,
        }
    }

    pub fn config(mut self, config: TrainConfig) -> Self {
        self.config = config;
        self
    }

    pub fn on_progress(mut self, callback: impl FnMut(Progress) + 'static) -> Self {
        self.on_progress = Some(Box::new(callback));
        self
    }

    /// Trains both nets and returns the assembled deployment model.
    pub fn train(self, bin_path: impl AsRef<Path>) -> Result<MergedDualEye<B::InnerBackend>, TrainError> {
        let Self { config, device, mut on_progress } = self;

        let frames = Arc::new(data::read_capture(bin_path).map_err(TrainError::Read)?);

        let mut report = |progress: Progress| {
            if let Some(callback) = &mut on_progress {
                callback(progress);
            }
        };

        let gaze = GazeTrainer::<B>::new(
            frames.clone(),
            &config,
            &device,
            Reporter::new(&mut report, Phase::Gaze, config.report_every, config.gaze_steps),
        )
        .fit()
        .valid();

        B::memory_cleanup(&device);

        let expr = ExprTrainer::<B>::new(
            frames,
            &config,
            &device,
            Reporter::new(&mut report, Phase::Expr, config.report_every, config.expr_steps),
        )
        .fit()
        .valid();

        Ok(MergedDualEye::new(DualTaskTower { gaze, expr }))
    }

    /// Trains and writes the deployment ONNX to `onnx_path`.
    pub fn train_to_onnx(
        self,
        bin_path: impl AsRef<Path>,
        onnx_path: impl AsRef<Path>,
    ) -> Result<(), TrainError> {
        let merged = self.train(bin_path)?;
        crate::export::export_onnx(&merged, onnx_path).map_err(TrainError::Export)
    }
}
