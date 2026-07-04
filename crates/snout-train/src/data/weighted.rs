//! Weighted composition of dataset pools.

use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{ComposedDataset, SamplerDataset};

/// Composes several dataset pools into one, drawing from each in proportion to its
/// weight (with replacement).
///
/// The mix's size is the sum of the pool lengths; each pool is sampled at that budget
/// scaled by its weight share, so the ratio between pools follows the weights. The
/// absolute size only affects sample diversity -- callers resample the result to the
/// number of items they actually need (e.g. `steps * batch_size`).
pub struct WeightedSampledDataset<D> {
    pools: Vec<(D, usize)>,
}

impl<D> WeightedSampledDataset<D> {
    /// `pools` pairs each source dataset with its (relative) weight; the weights need
    /// not sum to any particular value.
    pub fn new(pools: Vec<(D, usize)>) -> Self {
        Self { pools }
    }

    pub fn build<I>(self) -> ComposedDataset<SamplerDataset<D, I>>
    where
        D: Dataset<I>,
        I: Clone + Send + Sync,
    {
        let total: usize = self.pools.iter().map(|(d, _)| d.len()).sum();
        let weight_sum: usize = self.pools.iter().map(|(_, w)| *w).sum::<usize>().max(1);

        let pools: Vec<SamplerDataset<D, I>> = self
            .pools
            .into_iter()
            .filter(|(d, _)| d.len() > 0)
            .map(|(d, weight)| SamplerDataset::new(d, (total * weight / weight_sum).max(1)))
            .collect();

        ComposedDataset::new(pools)
    }
}
