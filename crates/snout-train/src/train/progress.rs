use std::fmt;

use burn::tensor::ElementConversion;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Which network is currently training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Gaze,
    Expr,
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Gaze => write!(f, "gaze"),
            Phase::Expr => write!(f, "expr"),
        }
    }
}

/// A training progress update.
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub phase: Phase,
    pub step: usize,
    pub total_steps: usize,
    pub loss: f32,
    /// Present on validation-cadence steps, carried alongside that step's loss; `None`
    /// otherwise.
    pub val: Option<ExprMetrics>,
}

/// Held-out expression validation metrics (Python `evaluate_expr`): mean squared error
/// over the four expression channels, plus per-channel Pearson correlation
/// (`[lid, widen, squint, brow]`).
#[derive(Debug, Clone, Copy)]
pub struct ExprMetrics {
    pub mse: f32,
    pub corr: [f32; 4],
}

/// Errors that can occur while training.
#[derive(Debug)]
pub enum TrainError {
    Read(String),
    Export(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainError::Read(e) => write!(f, "failed to read capture: {e}"),
            TrainError::Export(e) => write!(f, "failed to export onnx: {e}"),
        }
    }
}

impl std::error::Error for TrainError {}

/// Emits [`Progress`] updates at a fixed step cadence, syncing the loss scalar from
/// the GPU only on the steps it actually reports.
pub struct Reporter<'a> {
    callback: &'a mut dyn FnMut(Progress),
    phase: Phase,
    every: usize,
    total: usize,
}

impl<'a> Reporter<'a> {
    pub fn new(
        callback: &'a mut dyn FnMut(Progress),
        phase: Phase,
        every: usize,
        total: usize,
    ) -> Self {
        Self { callback, phase, every, total }
    }

    /// Emits a single [`Progress`] for `step` -- when it lands on the cadence (or is the
    /// final step) or when `val` is present -- carrying the loss and any validation
    /// metrics together. The loss scalar is synced from the GPU only when it reports.
    pub fn report<B: AutodiffBackend>(
        &mut self,
        loss: &Tensor<B, 1>,
        step: usize,
        total: usize,
        val: Option<ExprMetrics>,
    ) {
        let due = self.every > 0 && (step % self.every == 0 || step + 1 == total);
        if !due && val.is_none() {
            return;
        }
        let loss = loss.clone().into_scalar().elem::<f32>();
        (self.callback)(Progress {
            phase: self.phase,
            step,
            total_steps: self.total,
            loss,
            val,
        });
    }
}
