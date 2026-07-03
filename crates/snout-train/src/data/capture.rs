//! Reading and preprocessing capture `.bin` files into [`Frame`]s.

use std::{array::from_fn, path::Path};

use image::{GrayImage, ImageFormat};
use imageproc::contrast::equalize_histogram;
use snout_sample_file::{CaptureReader, FrameMeta};

use crate::data::label::{Expr, Gaze};
use crate::spec::{IMAGE_HEIGHT, IMAGE_WIDTH};

/// One preprocessed eye frame plus its (optional) supervision labels.
pub struct Frame {
    pub data: [f32; IMAGE_HEIGHT * IMAGE_WIDTH],
    pub expr: Option<Expr>,
    pub gaze: Option<Gaze>,
}

/// Loads and preprocesses frames from a bin file.
///
/// Returns a flat `Vec<Frame>` with pairs interleaved: left at even indices, right at
/// odd. Pairs where either side is blank/invalid are dropped entirely.
pub fn read_bin(path: impl AsRef<Path>) -> Result<Vec<Frame>, String> {
    let reader = CaptureReader::open(path.as_ref()).map_err(|e| e.to_string())?;

    let mut frames = Vec::new();

    for raw_frame in reader {
        let Ok(raw_frame) = raw_frame else { continue };

        if !raw_frame.meta.is_good_data() {
            continue;
        }

        let Some(left) = process_image(&raw_frame.jpeg_left, true) else { continue };
        let Some(right) = process_image(&raw_frame.jpeg_right, false) else { continue };

        let meta = &raw_frame.meta;
        frames.push(Frame {
            data: left,
            expr: compute_left_expr(meta),
            gaze: compute_left_gaze(meta),
        });
        frames.push(Frame {
            data: right,
            expr: compute_right_expr(meta),
            gaze: compute_right_gaze(meta),
        });
    }

    Ok(frames)
}

/// Canonical stage order for a session directory. Must stay in sync with the sampler's
/// `Stage::file_name` (snout `src/sample/sampler.rs`).
const SESSION_STAGES: [&str; 6] = ["gaze", "free-expr", "blink", "widen", "squint", "brow"];

/// Reads a capture from either a single `.bin` file or a session directory of per-stage
/// bins (`gaze.bin`, `free-expr.bin`, ...), concatenating present stages in canonical
/// order. Missing stage bins are skipped (graceful degradation).
pub fn read_capture(path: impl AsRef<Path>) -> Result<Vec<Frame>, String> {
    let path = path.as_ref();
    if !path.is_dir() {
        return read_bin(path);
    }

    let mut frames = Vec::new();
    for stage in SESSION_STAGES {
        let bin = path.join(format!("{stage}.bin"));
        if bin.exists() {
            frames.extend(read_bin(&bin)?);
        }
    }

    if frames.is_empty() {
        return Err(format!(
            "no stage bins ({}) found in session directory {}",
            SESSION_STAGES.join(", "),
            path.display()
        ));
    }

    Ok(frames)
}

fn gaze_normalize(angle_deg: f32) -> f32 {
    const RANGE: f32 = 45.0;
    ((angle_deg + RANGE) / (2.0 * RANGE)).clamp(0.0, 1.0)
}

fn compute_left_gaze(meta: &FrameMeta) -> Option<Gaze> {
    if !meta.is_gaze_data() {
        return None;
    }

    Some(Gaze {
        pitch: gaze_normalize(meta.left_eye_pitch),
        yaw: 1.0 - gaze_normalize(meta.left_eye_yaw),
    })
}

fn compute_left_expr(meta: &FrameMeta) -> Option<Expr> {
    if meta.is_expr_unlabeled() {
        return None;
    }

    Some(Expr {
        lid: 1.0 - meta.routine_left_lid.clamp(0.0, 1.0),
        widen: meta.routine_widen.clamp(0.0, 1.0),
        squint: meta.routine_squint.clamp(0.0, 1.0),
        brow: meta.routine_brow_angry.clamp(0.0, 1.0),
    })
}

fn compute_right_gaze(meta: &FrameMeta) -> Option<Gaze> {
    if !meta.is_gaze_data() {
        return None;
    }

    Some(Gaze {
        pitch: gaze_normalize(meta.right_eye_pitch),
        yaw: gaze_normalize(meta.right_eye_yaw),
    })
}

fn compute_right_expr(meta: &FrameMeta) -> Option<Expr> {
    if meta.is_expr_unlabeled() {
        return None;
    }

    Some(Expr {
        lid: 1.0 - meta.routine_right_lid.clamp(0.0, 1.0),
        widen: meta.routine_widen.clamp(0.0, 1.0),
        squint: meta.routine_squint.clamp(0.0, 1.0),
        brow: meta.routine_brow_angry.clamp(0.0, 1.0),
    })
}

fn process_image(jpeg: &[u8], flip: bool) -> Option<[f32; IMAGE_HEIGHT * IMAGE_WIDTH]> {
    fn is_blank(img: &GrayImage) -> bool {
        let pixels = img.as_raw();

        if pixels.is_empty() {
            return true;
        }

        let n = pixels.len() as f32;
        let mean = pixels.iter().map(|&p| p as f32).sum::<f32>() / n;
        let variance = pixels.iter().map(|&p| (p as f32 - mean).powi(2)).sum::<f32>() / n;

        variance.sqrt() < 2.0
    }

    fn resize(img: &GrayImage) -> GrayImage {
        let (width, height) = img.dimensions();

        if width == IMAGE_WIDTH as u32 && height == IMAGE_HEIGHT as u32 {
            return img.clone();
        }

        image::imageops::resize(
            img,
            IMAGE_WIDTH as u32,
            IMAGE_HEIGHT as u32,
            image::imageops::FilterType::Triangle,
        )
    }

    let img = image::load_from_memory_with_format(jpeg, ImageFormat::Jpeg)
        .ok()?
        .into_luma8();

    if is_blank(&img) {
        return None;
    }

    let img = resize(&img);
    let img = equalize_histogram(&img);
    let img = if flip { image::imageops::flip_horizontal(&img) } else { img };

    Some(from_fn(|i| img.as_raw()[i] as f32 / 255.0))
}
