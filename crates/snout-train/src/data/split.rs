//! Deterministic train/validation splitting of a dataset.

use std::sync::Arc;

use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{PartialDataset, RngSource, ShuffledDataset};

/// One side of a [`SplitDataset`]: a lazy window over the shared shuffled dataset.
pub type Partition<D, I> = PartialDataset<Arc<ShuffledDataset<D, I>>, I>;

/// Shuffles a dataset and splits it into two disjoint partitions.
///
/// The shuffle is a permutation, so every item lands in exactly one side -- the two
/// partitions never overlap. Used to carve a held-out validation set off the front of
/// the shuffled data, leaving the remainder for training.
pub struct SplitDataset<D> {
    dataset: D,
    rng: RngSource,
    val_frac: f64,
}

impl<D> SplitDataset<D> {
    /// `val_frac` is the fraction taken as the (first) validation partition.
    pub fn new(dataset: D, rng: impl Into<RngSource>, val_frac: f64) -> Self {
        Self { dataset, rng: rng.into(), val_frac }
    }

    /// Builds `(val, train)`: the first `val_frac` of the shuffled items, and the rest.
    pub fn build<I>(self) -> (Partition<D, I>, Partition<D, I>)
    where
        D: Dataset<I>,
        I: Clone + Send + Sync,
    {
        let shuffled = ShuffledDataset::new(self.dataset, self.rng);
        let n = shuffled.len();
        let val = ((n as f64) * self.val_frac) as usize;

        let shared = Arc::new(shuffled);
        let validation = PartialDataset::new(shared.clone(), 0, val);
        let train = PartialDataset::new(shared, val, n);
        (validation, train)
    }
}
