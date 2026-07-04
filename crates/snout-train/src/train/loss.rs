use burn::tensor::{Tensor, TensorData, backend::Backend};

/// Weight of the soft range-hinge term in the expression loss (Python `LAMBDA_RANGE`).
pub const LAMBDA_RANGE: f32 = 0.01;

/// Soft hinge keeping raw linear outputs near `[0, 1]`:
/// `mean(relu(-pred)^2 + relu(pred - 1)^2)`.
///
/// Only relevant for heads with unbounded (linear) outputs, e.g. the expression net.
pub fn range_hinge<B: Backend>(pred: Tensor<B, 2>) -> Tensor<B, 1> {
    let below = pred.clone().neg().clamp_min(0.0);
    let above = (pred - 1.0).clamp_min(0.0);
    (below.powf_scalar(2.0) + above.powf_scalar(2.0)).mean()
}

/// Per-channel deconfuse matrix `[driver][target]` (Python `EXPR_DECONFUSE`).
/// Expression channels co-occur physically (e.g. squinting also narrows the lid),
/// so when a driver expression is active we down-weight the loss on the targets it
/// confounds - the calibration labels for those are unreliable.
///
/// Channel order: `[EyeLid, EyeWiden, EyeSquint, EyeBrow]`.
const DECONFUSE: [[f32; 4]; 4] = [
    // target:  lid  widen squint brow
    [0.0, 0.0, 0.5, 0.7], // driver lid (blink): soft-mask squint + brow
    [0.0, 0.0, 0.0, 0.0], // driver widen: no confound
    [1.0, 0.0, 0.0, 1.0], // driver squint: mask lid + brow
    [0.7, 0.0, 0.7, 0.0], // driver brow: soft-mask lid + squint
];

/// Deconfuse-weighted MSE for expression regression.
///
/// The deconfuse matrix is a run-wide constant, so it is uploaded to the device once
/// (in [`DeconfuseLoss::new`]) and reused for every step.
pub struct DeconfuseLoss<B: Backend> {
    /// `[1, channels, channels]` deconfuse matrix `[driver][target]`.
    deconf: Tensor<B, 3>,
}

impl<B: Backend> DeconfuseLoss<B> {
    pub fn new(device: &B::Device) -> Self {
        let channels = DECONFUSE.len();
        let flat: Vec<f32> = DECONFUSE.iter().flatten().copied().collect();
        let deconf = Tensor::<B, 2>::from_data(TensorData::new(flat, [channels, channels]), device)
            .reshape([1, channels, channels]);
        Self { deconf }
    }

    /// `weight[b, j] = prod_i (1 - target[b, i] * DECONFUSE[i][j]).clamp_min(0)`, then
    /// `loss = sum(weight * (pred - target)^2) / sum(weight)`.
    pub fn forward(&self, pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch, channels] = target.dims();

        // factor[b, i, j] = clamp_min(1 - target[b, i] * deconf[i, j], 0)
        let drivers = target.clone().reshape([batch, channels, 1]);
        let factor = drivers.mul(self.deconf.clone()).neg().add_scalar(1.0).clamp_min(0.0);

        // weight[b, j] = product over the driver axis.
        let weight = factor.prod_dim(1).reshape([batch, channels]);

        let squared_error = (pred - target).powf_scalar(2.0);
        let numerator = (squared_error * weight.clone()).sum();
        let denominator = weight.sum().clamp_min(1.0);

        numerator / denominator
    }
}

/// Per-channel evenness weights `[lid, widen, squint, brow]` (Python `EXPR_EVEN_WEIGHTS`).
/// Widen is weighted highest — it's where per-eye asymmetry shows up most.
pub const EVEN_WEIGHTS: [f32; 4] = [0.3, 1.0, 0.3, 0.3];

/// Agree-gated L/R evenness loss for the (shared) expression tower.
///
/// Pulls the two eyes' predictions together *only where their labels agree*, so genuine
/// asymmetry (a wink, a one-sided squint) is left alone. Like [`DeconfuseLoss`], the
/// constant weight row is uploaded once and reused.
pub struct EvennessLoss<B: Backend> {
    /// `[1, channels]` per-channel weights.
    even_w: Tensor<B, 2>,
}

impl<B: Backend> EvennessLoss<B> {
    pub fn new(device: &B::Device) -> Self {
        let channels = EVEN_WEIGHTS.len();
        let even_w = Tensor::<B, 1>::from_data(TensorData::new(EVEN_WEIGHTS.to_vec(), [channels]), device)
            .reshape([1, channels]);
        Self { even_w }
    }

    /// `agree = (1 - |label_l - label_r|).clamp(0, 1)`, then
    /// `loss = sum(even_w * agree * (pred_l - pred_r)^2) / sum(agree)`.
    pub fn forward(
        &self,
        pred_l: Tensor<B, 2>,
        pred_r: Tensor<B, 2>,
        label_l: Tensor<B, 2>,
        label_r: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let agree = (label_l - label_r).abs().neg().add_scalar(1.0).clamp(0.0, 1.0);
        let diff_sq = (pred_l - pred_r).powf_scalar(2.0);
        let numerator = (self.even_w.clone() * agree.clone() * diff_sq).sum();
        numerator / agree.sum().clamp_min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    #[test]
    fn deconfuse_downweights_confounded_channels() {
        let device = Default::default();
        // One sample, squint fully active (driver squint masks lid + brow).
        let target = Tensor::<B, 2>::from_data([[0.0, 0.0, 1.0, 0.0]], &device);
        // Predict perfectly except a large error on the (masked) lid channel.
        let pred = Tensor::<B, 2>::from_data([[1.0, 0.0, 1.0, 0.0]], &device);

        // weight[lid] = (1 - squint*1.0) = 0 -> the lid error is fully ignored.
        // weight[widen]=1, weight[squint]=1, weight[brow]=(1-squint*1.0)=0.
        // numerator = 0 (only lid has error, weight 0). loss == 0.
        let loss = DeconfuseLoss::new(&device).forward(pred, target).into_scalar();
        assert!(loss.abs() < 1e-6, "expected ~0, got {loss}");
    }

    #[test]
    fn evenness_zero_when_predictions_equal() {
        let device = Default::default();
        let pred = Tensor::<B, 2>::from_data([[0.2, 0.5, 0.1, 0.0]], &device);
        let label = Tensor::<B, 2>::from_data([[0.0, 0.0, 0.0, 0.0]], &device);
        // identical L/R predictions -> no penalty regardless of agreement.
        let loss = EvennessLoss::new(&device)
            .forward(pred.clone(), pred, label.clone(), label)
            .into_scalar();
        assert!(loss.abs() < 1e-6, "expected ~0, got {loss}");
    }

    #[test]
    fn evenness_penalizes_disagreement_where_labels_agree() {
        let device = Default::default();
        // labels agree (both neutral); preds differ on widen -> penalized.
        let pred_l = Tensor::<B, 2>::from_data([[0.0, 1.0, 0.0, 0.0]], &device);
        let pred_r = Tensor::<B, 2>::from_data([[0.0, 0.0, 0.0, 0.0]], &device);
        let label = Tensor::<B, 2>::from_data([[0.0, 0.0, 0.0, 0.0]], &device);
        // agree=1 on all 4 channels (denom 4); widen weight 1.0, diff 1.0 -> numerator 1.0.
        let loss = EvennessLoss::new(&device)
            .forward(pred_l, pred_r, label.clone(), label)
            .into_scalar();
        assert!((loss - 0.25).abs() < 1e-6, "expected 0.25, got {loss}");
    }

    #[test]
    fn evenness_ignores_disagreement_where_labels_disagree() {
        let device = Default::default();
        // a genuine one-sided widen: labels disagree, so the gate zeroes that channel.
        let pred_l = Tensor::<B, 2>::from_data([[0.0, 1.0, 0.0, 0.0]], &device);
        let pred_r = Tensor::<B, 2>::from_data([[0.0, 0.0, 0.0, 0.0]], &device);
        let label_l = Tensor::<B, 2>::from_data([[0.0, 1.0, 0.0, 0.0]], &device);
        let label_r = Tensor::<B, 2>::from_data([[0.0, 0.0, 0.0, 0.0]], &device);
        let loss = EvennessLoss::new(&device)
            .forward(pred_l, pred_r, label_l, label_r)
            .into_scalar();
        assert!(loss.abs() < 1e-6, "expected ~0, got {loss}");
    }
}
