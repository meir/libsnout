/// Knobs for a full training run.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub gaze_steps: usize,
    pub expr_steps: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    /// How often (in steps) a progress update is emitted.
    pub report_every: usize,
    /// Fraction of capture instants held out (never trained on) for validation.
    pub val_frac: f64,
    /// Run expression validation every N expr steps (and at the end). `0` disables it.
    pub val_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            gaze_steps: 3200,
            expr_steps: 4000,
            batch_size: 64,
            num_workers: 4,
            seed: 1234,
            report_every: 50,
            val_frac: 0.1,
            val_every: 250,
        }
    }
}
