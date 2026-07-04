use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::capture::Frame;
use snout_sample_file::{CaptureWriter, FrameMeta, RawFrame, RoutineState};
use crate::sample::net::{Position, Routine};

const JPEG_QUALITY: u8 = 85;
const POSITION_FRESHNESS: Duration = Duration::from_millis(200);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Gaze,
    FreeExpr,
    Blink,
    Widen,
    Squint,
    Brow,
}

/// Per-frame expression labels stamped onto a captured frame.
#[derive(Default, Clone, Copy)]
struct Expr {
    lid: f32,
    brow_raise: f32,
    brow_angry: f32,
    widen: f32,
    squint: f32,
    dilate: f32,
}

impl Phase {
    pub fn from_routine(routine: Routine) -> Phase {
        match routine {
            Routine::Gaze(_) => Phase::Gaze,
            Routine::FreeExpr(_) => Phase::FreeExpr,
            Routine::Blink(_) => Phase::Blink,
            Routine::Widen(_) => Phase::Widen,
            Routine::Squint(_) => Phase::Squint,
            Routine::Brow(_) => Phase::Brow,
            _ => panic!("unsupported routine"),
        }
    }

    fn flags(&self) -> i32 {
        let mut flags = RoutineState::FLAG_GOOD_DATA
            | RoutineState::FLAG_IN_MOVEMENT
            | RoutineState::FLAG_VERSION_BIT1;
        if self.needs_position() {
            flags |= RoutineState::FLAG_GAZE_DATA;
        }
        if matches!(self, Phase::FreeExpr) {
            flags |= RoutineState::FLAG_EXPR_UNLABELED;
        }
        flags
    }

    fn needs_position(&self) -> bool {
        !matches!(self, Phase::Blink)
    }

    fn expr(&self) -> Expr {
        match self {
            Phase::Gaze | Phase::FreeExpr => Expr { lid: 1.0, ..Default::default() },
            Phase::Blink => Expr { lid: 0.0, ..Default::default() },
            Phase::Widen => Expr { lid: 1.0, widen: 1.0, ..Default::default() },
            Phase::Squint => Expr { lid: 1.0, squint: 1.0, ..Default::default() },
            Phase::Brow => Expr { lid: 1.0, brow_angry: 1.0, ..Default::default() },
        }
    }
}

pub struct FrameCollector {
    position: Option<(Position, Instant)>,
    frames: Vec<RawFrame>,
}

impl FrameCollector {
    pub fn new() -> Self {
        Self {
            position: None,
            frames: Vec::new(),
        }
    }

    pub fn set_position(&mut self, pos: Position) {
        self.position = Some((pos, Instant::now()));
    }

    pub fn add(&mut self, phase: Phase, left: &Frame, right: &Frame) {
        let position = if phase.needs_position() {
            let Some((pos, stamp)) = self.position else {
                return;
            };
            if stamp.elapsed() > POSITION_FRESHNESS {
                return;
            }
            Some(pos)
        } else {
            None
        };

        self.push(phase, left, right, position.as_ref());
    }

    pub fn write(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut writer = CaptureWriter::create(path)?;

        for frame in self.frames.drain(..) {
            writer.write_frame(&frame)?;
        }

        writer.flush()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    fn push(&mut self, phase: Phase, left: &Frame, right: &Frame, pos: Option<&Position>) {
        let time = timestamp_ms();
        let expr = phase.expr();

        // Flip them as baballonia does
        let jpeg_left = encode_jpeg(&right.image);
        let jpeg_right = encode_jpeg(&left.image);

        let meta = FrameMeta {
            routine_pitch: pos.map_or(0.0, |p| p.routine_pitch),
            routine_yaw: pos.map_or(0.0, |p| p.routine_yaw),
            routine_distance: pos.map_or(0.0, |p| p.routine_distance),
            routine_convergence: pos.map_or(0.0, |p| p.routine_convergence),
            fov_adjust_distance: pos.map_or(0.0, |p| p.fov_adjust_distance),
            left_eye_pitch: pos.map_or(0.0, |p| p.left_eye_pitch),
            left_eye_yaw: pos.map_or(0.0, |p| -p.left_eye_yaw),
            right_eye_pitch: pos.map_or(0.0, |p| p.right_eye_pitch),
            right_eye_yaw: pos.map_or(0.0, |p| -p.right_eye_yaw),
            routine_left_lid: expr.lid,
            routine_right_lid: expr.lid,
            routine_brow_raise: expr.brow_raise,
            routine_brow_angry: expr.brow_angry,
            routine_widen: expr.widen,
            routine_squint: expr.squint,
            routine_dilate: expr.dilate,
            timestamp: time,
            video_timestamp_left: time,
            video_timestamp_right: time,
            routine_state: RoutineState::from_raw(phase.flags()),
            jpeg_left_len: jpeg_left.len() as i32,
            jpeg_right_len: jpeg_right.len() as i32,
        };

        self.frames.push(RawFrame {
            meta,
            jpeg_left,
            jpeg_right,
        });
    }
}

fn encode_jpeg(image: &image::GrayImage) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
    encoder
        .encode(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::L8,
        )
        .expect("JPEG encoding failed");
    buf
}

fn timestamp_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
