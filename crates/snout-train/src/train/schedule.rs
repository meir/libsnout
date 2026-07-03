//! Learning-rate schedules.

/// Cosine decay factor in `[0, 1]`: 1.0 at `step = 0`, 0.0 at `step = total`.
fn cosine_factor(step: usize, total: usize) -> f64 {
    let progress = step as f64 / total.max(1) as f64;
    0.5 * (1.0 + (std::f64::consts::PI * progress.clamp(0.0, 1.0)).cos())
}

/// Linear warmup to `peak`, then cosine decay to `min` over the remaining steps.
#[derive(Debug, Clone, Copy)]
pub struct WarmupCosine {
    pub peak: f64,
    pub min: f64,
    pub warmup: usize,
    pub total: usize,
}

impl WarmupCosine {
    /// The gaze schedule (Python `train_gaze.py`).
    pub fn gaze(total: usize) -> Self {
        Self { peak: 1e-3, min: 1e-6, warmup: 500, total }
    }

    pub fn lr_at(&self, step: usize) -> f64 {
        if self.warmup > 0 && step < self.warmup {
            return self.peak * (step + 1) as f64 / self.warmup as f64;
        }
        let cos = cosine_factor(step - self.warmup, self.total - self.warmup);
        self.min + (self.peak - self.min) * cos
    }
}

/// One expression stage's `(head, backbone)` learning-rate schedule.
///
/// During the optional head-only warmup the backbone is held at `0.0` (frozen) and
/// the head at `warmup_head`; afterwards both decay by cosine from their peaks to
/// `min`. Mirrors the Python two-stage expression schedule (EXPR_STAGE_A/B).
#[derive(Debug, Clone, Copy)]
pub struct ExprStage {
    pub head: f64,
    pub backbone: f64,
    pub min: f64,
    pub total: usize,
    pub warmup: usize,
    pub warmup_head: f64,
}

impl ExprStage {
    /// Stage A: head-only warmup, then head 5e-4 / backbone 1e-4.
    pub const fn stage_a(total: usize) -> Self {
        Self { head: 5e-4, backbone: 1e-4, min: 1e-6, total, warmup: 200, warmup_head: 1e-3 }
    }

    /// Stage B: head 2e-4 / backbone 5e-5, cosine decay, no warmup.
    pub const fn stage_b(total: usize) -> Self {
        Self { head: 2e-4, backbone: 5e-5, min: 1e-6, total, warmup: 0, warmup_head: 0.0 }
    }

    pub fn lrs_at(&self, step: usize) -> (f64, f64) {
        if step < self.warmup {
            return (self.warmup_head, 0.0); // backbone frozen
        }
        let cos = cosine_factor(step - self.warmup, self.total - self.warmup);
        (
            self.min + (self.head - self.min) * cos,
            self.min + (self.backbone - self.min) * cos,
        )
    }
}

/// The full expression schedule as one continuous `(head, backbone)` curve over
/// `total` steps: phase A (head warmup, then the higher rates) covers the first
/// half, phase B (lower rates) the second. The LR *values* are identical to the
/// old per-stage trainer; expressing them as a single schedule lets one optimizer
/// (and its momentum) span the whole run instead of being rebuilt at the boundary.
#[derive(Debug, Clone, Copy)]
pub struct ExprSchedule {
    a: ExprStage,
    b: ExprStage,
    split: usize,
}

impl ExprSchedule {
    pub fn new(total: usize) -> Self {
        let a_steps = total / 2;
        Self {
            a: ExprStage::stage_a(a_steps),
            b: ExprStage::stage_b(total - a_steps),
            split: a_steps,
        }
    }

    /// `(head, backbone)` learning rates at a global `step`.
    pub fn lrs_at(&self, step: usize) -> (f64, f64) {
        if step < self.split {
            self.a.lrs_at(step)
        } else {
            self.b.lrs_at(step - self.split)
        }
    }
}
