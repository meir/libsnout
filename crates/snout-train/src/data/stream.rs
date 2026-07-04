//! High-level training streams: composing pools, augmentation and mixup into the
//! datasets the trainers feed to their data loaders.

use std::sync::Arc;

use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{ComposedDataset, MapperDataset, PartialDataset, RngSource, SamplerDataset, ShuffledDataset};

use crate::data::augment::{AugmentConfig, Augmenter, PairedAugmenter};
use crate::data::capture::Frame;
use crate::data::dataset::{PairedSampleDataset, PairedSampleItem, SampleDataset, SampleItem};
use crate::data::flatten::FlattenDataset;
use crate::data::mixup::MixupDataset;
use crate::data::samples;
use crate::data::split::SplitDataset;
use crate::data::weighted::WeightedSampledDataset;
use crate::train::TrainConfig;

// Gaze pool mix (percentages): 20% eyes-closed, the rest gaze-valid split into
// expression-during-gaze and neutral gaze (Python's 3-way stratified gaze sampler).
const GAZE_NEUTRAL_WEIGHT: usize = 32;
const GAZE_EXPR_WEIGHT: usize = 48;
const GAZE_CLOSED_WEIGHT: usize = 20;

// Mixup (Python EXPR_MIXUP_*): fraction of expr samples that are neutral<->active
// blends / general cross blends (the rest are plain).
const MIXUP_NEUTRAL_PROB: f64 = 0.5;
const MIXUP_CROSS_PROB: f64 = 0.15;

// Active-expression oversampling (Python `active_boost=4.0`). The calibration labels
// are ~binary, so an active frame's sampler weight is ~ `1 + EXPR_ACTIVE_BOOST` and a
// neutral frame's ~ 1. Sizing the two pools by that weight mass reproduces the
// sampler's active:neutral draw ratio using ComposedDataset + SamplerDataset.
const EXPR_ACTIVE_BOOST: usize = 4;

/// Paired-eye batch size for the evenness loss (Python `EXPR_EVEN_BATCH`).
pub const EXPR_EVEN_BATCH: usize = 32;

pub struct ExprData {
    pub a: ExprStream,
    pub b: PairedExprStream,
    pub c: EvalExprStream,
}

type S = Arc<FlattenDataset<T>>;
type T = PartialDataset<Arc<ShuffledDataset<PairedSampleDataset, PairedSampleItem>>, PairedSampleItem>;

impl ExprData {
    pub fn new(frames: &Arc<Vec<Frame>>, config: &TrainConfig) -> Self {
        let total = config.expr_steps;

        let pair_active = PairedSampleDataset::new(frames.clone(), Arc::new(samples::paired_active(frames)));
        let pair_neutral = PairedSampleDataset::new(frames.clone(), Arc::new(samples::paired_neutral(frames)));

        let (eval_active_pair, train_active_pair) = SplitDataset::new(pair_active, RngSource::Seed(config.seed), config.val_frac).build();
        let (eval_neutral_pair, train_neutral_pair) = SplitDataset::new(pair_neutral, RngSource::Seed(config.seed), config.val_frac).build();

        let train_active = Arc::new(FlattenDataset::new(train_active_pair.clone()));
        let train_neutral = Arc::new(FlattenDataset::new(train_neutral_pair.clone()));

        let eval_active = Arc::new(FlattenDataset::new(eval_active_pair));
        let eval_neutral = Arc::new(FlattenDataset::new(eval_neutral_pair));

        Self {
            a: ExprStream::new(train_active, train_neutral, total * config.batch_size),
            b: PairedExprStream::new(total * EXPR_EVEN_BATCH, train_active_pair, train_neutral_pair),
            c: EvalExprStream::new(eval_active, eval_neutral),
        }
    }
}

/// The gaze training stream: a stratified, lightly-augmented mix of the gaze pools.
pub struct GazeStream<'a> {
    frames: &'a Arc<Vec<Frame>>,
    total_items: usize,
}

impl<'a> GazeStream<'a> {
    pub fn new(frames: &'a Arc<Vec<Frame>>, total_items: usize) -> Self {
        Self { frames, total_items }
    }

    pub fn build(self) -> impl Dataset<SampleItem> + use<> {
        let base = WeightedSampledDataset::new(vec![
            (SampleDataset::new(self.frames.clone(), Arc::new(samples::neutral_gaze(self.frames))), GAZE_NEUTRAL_WEIGHT),
            (SampleDataset::new(self.frames.clone(), Arc::new(samples::expr_gaze(self.frames))), GAZE_EXPR_WEIGHT),
            (SampleDataset::new(self.frames.clone(), Arc::new(samples::closed(self.frames))), GAZE_CLOSED_WEIGHT),
        ]).build();

        let base = SamplerDataset::new(base, self.total_items);
        let base = MapperDataset::new(base, Augmenter::new(AugmentConfig::gaze()));

        base
    }
}

/// The expression training stream: an active-boosted base of augmented samples, with
/// mixup blending toward intermediate expression intensities.
pub struct ExprStream {
    total_items: usize,
    active: S,
    neutral: S,
}

impl ExprStream {
    pub fn new(active: S, neutral: S, total_items: usize) -> Self {
        Self { total_items, active, neutral }
    }

    pub fn build(self) -> impl Dataset<SampleItem> + use<> {
        let active_mass = self.active.len() * (1 + EXPR_ACTIVE_BOOST);
        let neutral_mass = self.neutral.len();
        let mass = (active_mass + neutral_mass).max(1);

        let base = WeightedSampledDataset::new(vec![
            (self.active.clone(), 100 * active_mass / mass),
            (self.neutral.clone(), 100 * neutral_mass / mass),
        ]).build();

        let base = SamplerDataset::new(base, self.total_items);
        let base = MapperDataset::new(base, Augmenter::new(AugmentConfig::default()));

        MixupDataset::new(
            base,
            MapperDataset::new(self.neutral, Augmenter::new(AugmentConfig::default())),
            MapperDataset::new(self.active, Augmenter::new(AugmentConfig::default())),
            MIXUP_NEUTRAL_PROB,
            MIXUP_CROSS_PROB,
        )
    }
}

/// The paired-eye expression stream: both eyes of the same capture instant, each
/// augmented independently, for the evenness loss. Real frames only (no mixup) — the
/// L/R pairing must survive.
pub struct PairedExprStream {
    total_items: usize,
    active: T,
    neutral: T,
}

impl PairedExprStream {
    pub fn new(total_items: usize, active: T, neutral: T) -> Self {
        Self { total_items, active, neutral }
    }

    /// `None` when the capture has no valid L/R pairs (single-eye data) — the caller
    /// then skips the evenness term.
    pub fn build(self) -> Option<impl Dataset<PairedSampleItem> + use<>> {
        if self.active.is_empty() && self.neutral.is_empty() {
            return None;
        }

        // Active-boost the paired sampler with the same mass weighting as the main
        // expression stream (Python `paired_ds.sampler_weights(active_boost)`).
        let active_mass = self.active.len() * (1 + EXPR_ACTIVE_BOOST);
        let neutral_mass = self.neutral.len();
        let mass = (active_mass + neutral_mass).max(1);

        let base = WeightedSampledDataset::new(vec![
            (self.active.clone(), 100 * active_mass / mass),
            (self.neutral.clone(), 100 * neutral_mass / mass),
        ]).build();

        let base = SamplerDataset::new(base, self.total_items);
        let base = MapperDataset::new(base, PairedAugmenter::new(AugmentConfig::default()));

        Some(base)
    }
}

/// The held-out expression evaluation stream: the validation partitions of the active
/// and neutral pools, concatenated raw -- no augmentation, mixup, or oversampling -- so
/// a single loader pass scores every held-out sample exactly once.
pub struct EvalExprStream {
    active: S,
    neutral: S,
}

impl EvalExprStream {
    pub fn new(active: S, neutral: S) -> Self {
        Self { active, neutral }
    }

    pub fn build(self) -> impl Dataset<SampleItem> + use<> {
        ComposedDataset::new(vec![self.active, self.neutral])
    }
}
