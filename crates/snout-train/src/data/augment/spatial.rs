use image::Luma;
use imageproc::geometric_transformations::{Interpolation, Projection, warp};
use rand::Rng;

use super::FloatImage;

#[derive(Debug, Clone, Copy)]
pub struct SpatialParams {
    /// Maximum per-axis translation, in pixels. Sampled inclusively.
    pub max_shift: i32,
    /// Maximum rotation magnitude in **degrees**.
    pub max_rotation_deg: f32,
    /// Scale jitter magnitude.
    pub max_scale: f32,
}

impl Default for SpatialParams {
    fn default() -> Self {
        Self {
            max_shift: 10,
            max_rotation_deg: 10.0,
            max_scale: 0.15,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialPlan {
    pub shift_x: i32,
    pub shift_y: i32,
    pub angle_deg: f32,
    pub scale: f32,
}

impl SpatialPlan {
    pub fn sample<R: Rng + ?Sized>(rng: &mut R, params: &SpatialParams) -> Self {
        let shift_x = rng.gen_range(-params.max_shift..=params.max_shift);
        let shift_y = rng.gen_range(-params.max_shift..=params.max_shift);
        let angle_deg = rng.gen_range(-params.max_rotation_deg..params.max_rotation_deg);
        let scale = 1.0 + rng.gen_range(-params.max_scale..params.max_scale);
        Self {
            shift_x,
            shift_y,
            angle_deg,
            scale,
        }
    }

    fn projection(&self, width: u32, height: u32) -> Projection {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let theta = -self.angle_deg.to_radians();

        Projection::translate(-cx, -cy)
            .and_then(Projection::scale(self.scale, self.scale))
            .and_then(Projection::rotate(theta))
            .and_then(Projection::translate(
                cx + self.shift_x as f32,
                cy + self.shift_y as f32,
            ))
    }

    pub fn apply(&self, img: &FloatImage) -> FloatImage {
        let (w, h) = img.dimensions();
        let projection = self.projection(w, h);
        // NOTE: black border fill. The Python uses BORDER_REPLICATE; imageproc's `warp`
        // only supports a fixed fill color, so out-of-frame pixels become 0.0 here.
        warp(img, &projection, Interpolation::Bilinear, Luma([0.0f32]))
    }
}
