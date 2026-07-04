//! Collating items into device tensors.

use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::data::dataset::{PairedSampleItem, SampleItem};
use crate::data::label::{Expr, Gaze};
use crate::spec::{EXPR_OUTPUTS, GAZE_OUTPUTS, IMAGE_HEIGHT, IMAGE_WIDTH, PIXELS_PER_FRAME, TEMPORAL_DEPTH};

/// Gaze fill for samples without a gaze label (center gaze).
const NO_GAZE: [f32; GAZE_OUTPUTS] = [0.5; GAZE_OUTPUTS];
/// Expression fill for samples without an expression label.
const NO_EXPR: [f32; EXPR_OUTPUTS] = [0.0; EXPR_OUTPUTS];

fn image_tensor<B: Backend>(pixels: Vec<f32>, batch: usize, device: &B::Device) -> Tensor<B, 4> {
    Tensor::from_data(
        TensorData::new(pixels, [batch, TEMPORAL_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]),
        device,
    )
}

fn label_tensor<B: Backend>(values: Vec<f32>, batch: usize, width: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values, [batch, width]), device)
}

#[derive(Clone, Debug)]
pub struct SampleBatch<B: Backend> {
    /// `[batch, TEMPORAL_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]`
    pub inputs: Tensor<B, 4>,
    /// `[batch, 2]` - pitch, yaw (center where gaze is None).
    pub gaze: Tensor<B, 2>,
    /// `[batch, 4]` - lid, widen, squint, brow (zeros where expr is None).
    pub expr: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct SampleBatcher;

impl<B: Backend> Batcher<B, SampleItem, SampleBatch<B>> for SampleBatcher {
    fn batch(&self, items: Vec<SampleItem>, device: &B::Device) -> SampleBatch<B> {
        let batch = items.len();
        let mut images = Vec::with_capacity(batch * TEMPORAL_DEPTH * PIXELS_PER_FRAME);
        let mut gaze = Vec::with_capacity(batch * GAZE_OUTPUTS);
        let mut expr = Vec::with_capacity(batch * EXPR_OUTPUTS);

        for item in items {
            images.extend_from_slice(&item.image);
            gaze.extend(item.gaze.map_or(NO_GAZE, Gaze::to_array));
            expr.extend(item.expr.map_or(NO_EXPR, Expr::to_array));
        }

        SampleBatch {
            inputs: image_tensor(images, batch, device),
            gaze: label_tensor(gaze, batch, GAZE_OUTPUTS, device),
            expr: label_tensor(expr, batch, EXPR_OUTPUTS, device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PairedSampleBatch<B: Backend> {
    /// `[batch, TEMPORAL_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]`
    pub left_inputs: Tensor<B, 4>,
    /// `[batch, TEMPORAL_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH]`
    pub right_inputs: Tensor<B, 4>,
    /// `[batch, 4]`
    pub left_expr: Tensor<B, 2>,
    /// `[batch, 4]`
    pub right_expr: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct PairedSampleBatcher;

impl<B: Backend> Batcher<B, PairedSampleItem, PairedSampleBatch<B>> for PairedSampleBatcher {
    fn batch(&self, items: Vec<PairedSampleItem>, device: &B::Device) -> PairedSampleBatch<B> {
        let batch = items.len();
        let mut left = Vec::with_capacity(batch * TEMPORAL_DEPTH * PIXELS_PER_FRAME);
        let mut right = Vec::with_capacity(batch * TEMPORAL_DEPTH * PIXELS_PER_FRAME);
        let mut left_expr = Vec::with_capacity(batch * EXPR_OUTPUTS);
        let mut right_expr = Vec::with_capacity(batch * EXPR_OUTPUTS);

        for item in items {
            left.extend_from_slice(&item.left_image);
            right.extend_from_slice(&item.right_image);
            left_expr.extend(item.left_expr.map_or(NO_EXPR, Expr::to_array));
            right_expr.extend(item.right_expr.map_or(NO_EXPR, Expr::to_array));
        }

        PairedSampleBatch {
            left_inputs: image_tensor(left, batch, device),
            right_inputs: image_tensor(right, batch, device),
            left_expr: label_tensor(left_expr, batch, EXPR_OUTPUTS, device),
            right_expr: label_tensor(right_expr, batch, EXPR_OUTPUTS, device),
        }
    }
}
