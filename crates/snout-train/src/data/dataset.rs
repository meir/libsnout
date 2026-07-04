//! Datasets that yield temporal-stack items from the frame pool.

use std::sync::Arc;

use burn::data::dataset::Dataset;

use crate::data::capture::Frame;
use crate::data::label::{Expr, Gaze};
use crate::data::samples::Sample;
use crate::spec::{PIXELS_PER_FRAME, TEMPORAL_DEPTH};

/// Concatenates a temporal stack's frame pixels into one flat CHW buffer (newest first).
pub(crate) fn stack_pixels(frames: &[Frame], sample: &Sample) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(TEMPORAL_DEPTH * PIXELS_PER_FRAME);
    for &idx in sample {
        pixels.extend_from_slice(&frames[idx].data);
    }
    pixels
}

#[derive(Clone, Debug)]
pub struct SampleItem {
    /// Flattened temporal stack: `[TEMPORAL_DEPTH * 128 * 128]` in CHW order (newest first).
    pub image: Vec<f32>,
    /// Expression label from the newest frame (if supervised).
    pub expr: Option<Expr>,
    /// Gaze label from the newest frame (if gaze-valid).
    pub gaze: Option<Gaze>,
}

#[derive(Clone)]
pub struct SampleDataset {
    frames: Arc<Vec<Frame>>,
    samples: Arc<Vec<Sample>>,
}

impl SampleDataset {
    pub fn new(frames: Arc<Vec<Frame>>, samples: Arc<Vec<Sample>>) -> Self {
        Self { frames, samples }
    }
}

impl Dataset<SampleItem> for SampleDataset {
    fn get(&self, index: usize) -> Option<SampleItem> {
        let sample = self.samples.get(index)?;
        let newest = &self.frames[sample[0]];

        Some(SampleItem {
            image: stack_pixels(&self.frames, sample),
            expr: newest.expr,
            gaze: newest.gaze,
        })
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

#[derive(Clone, Debug)]
pub struct PairedSampleItem {
    /// Left eye temporal stack: `[TEMPORAL_DEPTH * 128 * 128]`.
    pub left_image: Vec<f32>,
    /// Right eye temporal stack: `[TEMPORAL_DEPTH * 128 * 128]`.
    pub right_image: Vec<f32>,
    /// Left eye expression label.
    pub left_expr: Option<Expr>,
    /// Right eye expression label.
    pub right_expr: Option<Expr>,
}

#[derive(Clone)]
pub struct PairedSampleDataset {
    frames: Arc<Vec<Frame>>,
    samples: Arc<Vec<[Sample; 2]>>,
}

impl PairedSampleDataset {
    pub fn new(frames: Arc<Vec<Frame>>, samples: Arc<Vec<[Sample; 2]>>) -> Self {
        Self { frames, samples }
    }
}

impl Dataset<PairedSampleItem> for PairedSampleDataset {
    fn get(&self, index: usize) -> Option<PairedSampleItem> {
        let [left, right] = self.samples.get(index)?;

        Some(PairedSampleItem {
            left_image: stack_pixels(&self.frames, left),
            right_image: stack_pixels(&self.frames, right),
            left_expr: self.frames[left[0]].expr,
            right_expr: self.frames[right[0]].expr,
        })
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}
