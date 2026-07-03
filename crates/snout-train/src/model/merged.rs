use burn::prelude::*;
use burn::tensor::Int;

use crate::model::tower::DualTaskTower;
use crate::spec::EXPR_OUTPUTS;

/// Channels fed to the tower un-flipped (canonical eye). Python `EXPORT_NOFLIP_CH`
/// for `SWAP_INPUT_EYES = true`.
const NOFLIP_CHANNELS: [i32; 4] = [1, 3, 5, 7];
/// Channels horizontally flipped into canonical space. Python `EXPORT_FLIP_CH`.
const FLIP_CHANNELS: [i32; 4] = [0, 2, 4, 6];

/// Deployment model: processes both eyes of an interleaved temporal stack and
/// emits the 12-channel blendshape vector.
///
/// Input `[B, 8, 128, 128]` (channels interleaved L/R across `t0..t3`, gray [0,1]):
/// - the non-flipped eye is fed directly;
/// - the other eye is horizontally flipped into canonical space.
///
/// Output `[B, 12]` = `[right(EyeY, EyeX, EyeLid, EyeWiden, EyeSquint, EyeBrow), left(..)]`,
/// clamped to `[0, 1]`. The right eye's yaw is mirrored (`1 - yaw`).
#[derive(Module, Debug)]
pub struct MergedDualEye<B: Backend> {
    pub tower: DualTaskTower<B>,
}

impl<B: Backend> MergedDualEye<B> {
    pub fn new(tower: DualTaskTower<B>) -> Self {
        Self { tower }
    }

    /// `[B, 8, 128, 128] -> [B, 12]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let device = x.device();
        let noflip_idx = Tensor::<B, 1, Int>::from_data(NOFLIP_CHANNELS, &device);
        let flip_idx = Tensor::<B, 1, Int>::from_data(FLIP_CHANNELS, &device);

        let noflip = x.clone().select(1, noflip_idx);
        // Mirror the other eye into canonical space (flip the width dimension).
        let flipped = x.select(1, flip_idx).flip([3]);

        let out_left = self.tower.forward(noflip);
        let out_right = mirror_yaw(self.tower.forward(flipped));

        Tensor::cat(vec![out_right, out_left], 1).clamp(0.0, 1.0)
    }
}

/// Mirrors the yaw channel (index 1) of a per-eye output: `yaw -> 1 - yaw`.
fn mirror_yaw<B: Backend>(out: Tensor<B, 2>) -> Tensor<B, 2> {
    let pitch = out.clone().narrow(1, 0, 1);
    let yaw = out.clone().narrow(1, 1, 1).neg().add_scalar(1.0);
    let rest = out.narrow(1, 2, EXPR_OUTPUTS);
    Tensor::cat(vec![pitch, yaw, rest], 1)
}
