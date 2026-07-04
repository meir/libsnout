mod blur;
mod intensity;
mod spatial;

use burn::data::dataset::transform::Mapper;
use image::{ImageBuffer, Luma};
use rand::Rng;

use crate::data::dataset::{PairedSampleItem, SampleItem};
use crate::spec::{IMAGE_HEIGHT, IMAGE_WIDTH, PIXELS_PER_FRAME};
use blur::{BlurParams, BlurPlan};
use intensity::{IntensityParams, IntensityPlan};
use spatial::{SpatialParams, SpatialPlan};

pub(super) type FloatImage = ImageBuffer<Luma<f32>, Vec<f32>>;

#[derive(Debug, Clone, Copy)]
pub struct AugmentConfig {
    pub spatial_probability: f32,
    pub spatial: SpatialParams,
    pub intensity_probability: f32,
    pub intensity: IntensityParams,
    pub blur_probability: f32,
    pub blur: BlurParams,
}

impl Default for AugmentConfig {
    /// Strong augmentation, matching the Python expression stack.
    fn default() -> Self {
        Self {
            spatial_probability: 1.0,
            spatial: SpatialParams::default(),
            intensity_probability: 1.0,
            intensity: IntensityParams::default(),
            blur_probability: 0.5,
            blur: BlurParams::default(),
        }
    }
}

impl AugmentConfig {
    /// Lighter augmentation, matching the Python gaze stack (`_aug_stack_gaze`).
    ///
    /// Each transform is gated by its own probability, and the magnitudes differ
    /// from the expression stack: wider translation, stronger contrast, ksize-derived
    /// blur, and no gamma.
    pub fn gaze() -> Self {
        Self {
            spatial_probability: 0.3,
            // angle ±12°, tx/ty ±0.172·128 ≈ ±22px, scale 0.9-1.1.
            spatial: SpatialParams {
                max_shift: 22,
                max_rotation_deg: 12.0,
                max_scale: 0.10,
            },
            intensity_probability: 0.4,
            // brightness ±0.2, contrast 0.4-1.6, gamma disabled.
            intensity: IntensityParams {
                brightness_range: 0.2,
                contrast_range: 0.6,
                gamma_min: 1.0,
                gamma_max: 1.0,
            },
            blur_probability: 0.3,
            // Python samples a kernel size in [3, 15]; cv2's default sigma for those
            // kernels spans ≈0.8-2.6, approximated here as a continuous range.
            blur: BlurParams {
                sigma_min: 0.8,
                sigma_max: 2.6,
            },
        }
    }

    /// Applies the augmentation to a flattened temporal stack.
    /// One plan per transform is sampled and shared across all frames in the stack.
    fn apply<R: Rng + ?Sized>(&self, image: &[f32], rng: &mut R) -> Vec<f32> {
        let mut frames = split_frames(image);

        if rng.gen_bool(self.spatial_probability.clamp(0.0, 1.0) as f64) {
            let plan = SpatialPlan::sample(rng, &self.spatial);
            for f in frames.iter_mut() {
                *f = plan.apply(f);
            }
        }

        if rng.gen_bool(self.intensity_probability.clamp(0.0, 1.0) as f64) {
            let plan = IntensityPlan::sample(rng, &self.intensity);
            for f in frames.iter_mut() {
                plan.apply_in_place(f);
            }
        }

        if rng.gen_bool(self.blur_probability.clamp(0.0, 1.0) as f64) {
            let plan = BlurPlan::sample(rng, &self.blur);
            for f in frames.iter_mut() {
                *f = plan.apply(f);
            }
        }

        flatten_frames(&frames)
    }
}

fn split_frames(image: &[f32]) -> Vec<FloatImage> {
    image
        .chunks(PIXELS_PER_FRAME)
        .map(|chunk| {
            ImageBuffer::from_raw(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, chunk.to_vec())
                .expect("frame chunk has the correct length")
        })
        .collect()
}

fn flatten_frames(frames: &[FloatImage]) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames.len() * PIXELS_PER_FRAME);
    for f in frames {
        out.extend_from_slice(f.as_raw());
    }
    out
}

#[derive(Clone)]
pub struct Augmenter {
    config: AugmentConfig,
}

impl Augmenter {
    pub fn new(config: AugmentConfig) -> Self {
        Self { config }
    }
}

impl Mapper<SampleItem, SampleItem> for Augmenter {
    fn map(&self, item: &SampleItem) -> SampleItem {
        let mut rng = rand::thread_rng();
        SampleItem {
            image: self.config.apply(&item.image, &mut rng),
            expr: item.expr,
            gaze: item.gaze,
        }
    }
}

#[derive(Clone)]
pub struct PairedAugmenter {
    config: AugmentConfig,
}

impl PairedAugmenter {
    pub fn new(config: AugmentConfig) -> Self {
        Self { config }
    }
}

impl Mapper<PairedSampleItem, PairedSampleItem> for PairedAugmenter {
    fn map(&self, item: &PairedSampleItem) -> PairedSampleItem {
        let mut rng = rand::thread_rng();
        // Independent augmentation per eye: they are different images, and the
        // evenness loss benefits from light L/R augmentation inconsistency.
        PairedSampleItem {
            left_image: self.config.apply(&item.left_image, &mut rng),
            right_image: self.config.apply(&item.right_image, &mut rng),
            left_expr: item.left_expr,
            right_expr: item.right_expr,
        }
    }
}
