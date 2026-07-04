//! Flattening a paired-sample dataset into a stream of single samples.

use burn::data::dataset::Dataset;

use crate::data::dataset::{PairedSampleItem, SampleItem};

/// Turns a dataset of L/R pairs into a dataset of single samples: each pair yields two
/// items (left then right), so the length doubles. Paired items carry no gaze label, so
/// the flattened samples have `gaze: None`.
pub struct FlattenDataset<D> {
    inner: D,
}

impl<D> FlattenDataset<D> {
    pub fn new(inner: D) -> Self {
        Self { inner }
    }
}

impl<D> Dataset<SampleItem> for FlattenDataset<D>
where
    D: Dataset<PairedSampleItem>,
{
    fn get(&self, index: usize) -> Option<SampleItem> {
        let pair = self.inner.get(index / 2)?;
        Some(if index % 2 == 0 {
            SampleItem { image: pair.left_image, expr: pair.left_expr, gaze: None }
        } else {
            SampleItem { image: pair.right_image, expr: pair.right_expr, gaze: None }
        })
    }

    fn len(&self) -> usize {
        self.inner.len() * 2
    }
}
