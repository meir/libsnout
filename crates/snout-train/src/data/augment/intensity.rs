use rand::Rng;

use super::FloatImage;

#[derive(Debug, Clone, Copy)]
pub struct IntensityParams {
    /// Magnitude of the brightness offset.
    pub brightness_range: f32,
    /// Magnitude of the contrast jitter.
    pub contrast_range: f32,
    /// Inclusive gamma range `[min, max]`, sampled uniformly and applied as
    /// `x.powf(gamma)` after the brightness/contrast clamp (Python `x ** gamma`).
    /// Set both bounds to `1.0` to disable gamma (e.g. the gaze stack).
    pub gamma_min: f32,
    pub gamma_max: f32,
}

impl Default for IntensityParams {
    fn default() -> Self {
        Self {
            brightness_range: 0.3,
            contrast_range: 0.3,
            gamma_min: 0.6,
            gamma_max: 1.6,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntensityPlan {
    pub brightness: f32,
    pub contrast: f32,
    pub gamma: f32,
}

impl IntensityPlan {
    pub fn sample<R: Rng + ?Sized>(rng: &mut R, params: &IntensityParams) -> Self {
        let brightness = rng.gen_range(-params.brightness_range..params.brightness_range);
        let contrast = 1.0 + rng.gen_range(-params.contrast_range..params.contrast_range);
        let gamma = if params.gamma_min < params.gamma_max {
            rng.gen_range(params.gamma_min..params.gamma_max)
        } else {
            params.gamma_min
        };
        Self {
            brightness,
            contrast,
            gamma,
        }
    }

    pub fn apply_in_place(&self, img: &mut FloatImage) {
        for v in img.as_mut() {
            let x = (*v * self.contrast + self.brightness).clamp(0.0, 1.0);
            // Gamma is applied to the clamped [0, 1] value, matching Python's
            // `np.clip(x * contrast + bright, 0, 1) ** gamma`.
            *v = if self.gamma == 1.0 { x } else { x.powf(self.gamma) };
        }
    }
}
