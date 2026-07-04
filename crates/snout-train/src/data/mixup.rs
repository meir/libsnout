use burn::data::dataset::Dataset;
use rand::Rng;

use crate::data::dataset::SampleItem;
use crate::data::label::Expr;

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Blends pairs of (already-augmented) samples to synthesize intermediate expression
/// intensities, since the calibration labels are binary. Per `get`:
/// - `neutral_prob`: blend a random `neutral` sample with a random `active` one,
/// - `cross_prob`: blend two random samples from the `base` stream,
/// - otherwise: return the indexed `base` sample unchanged.
///
/// `base` is the (already active-boosted) stream that drives the dataset length and
/// the plain / cross-blend samples; `neutral` and `active` are standalone pools used
/// only as blend sources.
///
/// Blended samples carry no gaze label (the blend has no meaningful gaze).
#[derive(Clone)]
pub struct MixupDataset<Base, Pool> {
    base: Base,
    neutral: Pool,
    active: Pool,
    neutral_prob: f64,
    cross_prob: f64,
}

impl<Base, Pool> MixupDataset<Base, Pool> {
    pub fn new(base: Base, neutral: Pool, active: Pool, neutral_prob: f64, cross_prob: f64) -> Self {
        Self {
            base,
            neutral,
            active,
            neutral_prob,
            cross_prob,
        }
    }
}

impl<Base, Pool> Dataset<SampleItem> for MixupDataset<Base, Pool>
where
    Base: Dataset<SampleItem>,
    Pool: Dataset<SampleItem>,
{
    fn get(&self, index: usize) -> Option<SampleItem> {
        let mut rng = rand::thread_rng();
        let roll: f64 = rng.r#gen();

        if roll < self.neutral_prob && !self.neutral.is_empty() && !self.active.is_empty() {
            let neutral = self.neutral.get(rng.gen_range(0..self.neutral.len()))?;
            let active = self.active.get(rng.gen_range(0..self.active.len()))?;
            Some(blend(neutral, active, rng.r#gen::<f32>()))
        } else if roll < self.neutral_prob + self.cross_prob && self.base.len() > 1 {
            let a = self.base.get(rng.gen_range(0..self.base.len()))?;
            let b = self.base.get(rng.gen_range(0..self.base.len()))?;
            Some(blend(a, b, rng.r#gen::<f32>()))
        } else {
            self.base.get(index)
        }
    }

    fn len(&self) -> usize {
        self.base.len()
    }
}

/// Linearly blends two samples: `out = (1 - lam) * a + lam * b` on pixels and
/// expression labels. Gaze is dropped.
fn blend(a: SampleItem, b: SampleItem, lam: f32) -> SampleItem {
    let image = a
        .image
        .iter()
        .zip(&b.image)
        .map(|(&x, &y)| lerp(x, y, lam))
        .collect();

    let expr = match (a.expr, b.expr) {
        (Some(ea), Some(eb)) => {
            let (ea, eb) = (ea.to_array(), eb.to_array());
            Some(Expr::from_array(std::array::from_fn(|i| lerp(ea[i], eb[i], lam))))
        }
        (a, b) => a.or(b),
    };

    SampleItem { image, expr, gaze: None }
}
