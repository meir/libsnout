use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::capture::Frame;
use crate::sample::file::{CaptureWriter, FrameMeta, RawFrame, RoutineState};
use crate::sample::net::Position;

const JPEG_QUALITY: u8 = 85;
const POSITION_FRESHNESS: Duration = Duration::from_millis(200);

pub enum Phase {
    Gaze,
    Blink,
}

impl Phase {
    fn flags(&self) -> i32 {
        match self {
            Phase::Gaze => {
                RoutineState::FLAG_GOOD_DATA
                    | RoutineState::FLAG_IN_MOVEMENT
                    | RoutineState::FLAG_VERSION_BIT1
                    | RoutineState::FLAG_GAZE_DATA
            }
            Phase::Blink => {
                RoutineState::FLAG_GOOD_DATA
                    | RoutineState::FLAG_IN_MOVEMENT
                    | RoutineState::FLAG_VERSION_BIT1
            }
        }
    }

    fn needs_position(&self) -> bool {
        match self {
            Phase::Gaze => true,
            Phase::Blink => false,
        }
    }

    fn lid(&self) -> f32 {
        match self {
            Phase::Gaze => 1.0,
            Phase::Blink => 0.0,
        }
    }
}

pub struct FrameCollector {
    phase: Phase,
    position: Option<(Position, Instant)>,
    frames: Vec<RawFrame>,
}

impl FrameCollector {
    pub fn new(phase: Phase) -> Self {
        Self {
            phase,
            position: None,
            frames: Vec::new(),
        }
    }

    pub fn set_position(&mut self, pos: Position) {
        self.position = Some((pos, Instant::now()));
    }

    pub fn add(&mut self, left: &Frame, right: &Frame) {
        if self.phase.needs_position() {
            let Some((pos, stamp)) = self.position.clone() else {
                return;
            };
            if stamp.elapsed() > POSITION_FRESHNESS {
                return;
            }
            self.add_with_position(left, right, &pos);
        } else {
            self.add_without_position(left, right);
        }
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

    fn add_with_position(&mut self, left: &Frame, right: &Frame, pos: &Position) {
        let time = timestamp_ms();
        let meta = FrameMeta {
            routine_pitch: pos.routine_pitch,
            routine_yaw: pos.routine_yaw,
            routine_distance: pos.routine_distance,
            routine_convergence: pos.routine_convergence,
            fov_adjust_distance: pos.fov_adjust_distance,
            left_eye_pitch: pos.left_eye_pitch,
            left_eye_yaw: -pos.left_eye_yaw,
            right_eye_pitch: pos.right_eye_pitch,
            right_eye_yaw: -pos.right_eye_yaw,
            routine_left_lid: self.phase.lid(),
            routine_right_lid: self.phase.lid(),
            routine_brow_raise: 0.0,
            routine_brow_angry: 0.0,
            routine_widen: 0.0,
            routine_squint: 0.0,
            routine_dilate: 0.0,
            timestamp: time,
            video_timestamp_left: time,
            video_timestamp_right: time,
            routine_state: RoutineState::from_raw(self.phase.flags()),
            jpeg_left_len: 0,
            jpeg_right_len: 0,
        };
        self.push_frame(meta, left, right);
    }

    fn add_without_position(&mut self, left: &Frame, right: &Frame) {
        let time = timestamp_ms();
        let meta = FrameMeta {
            routine_pitch: 0.0,
            routine_yaw: 0.0,
            routine_distance: 0.0,
            routine_convergence: 0.0,
            fov_adjust_distance: 0.0,
            left_eye_pitch: 0.0,
            left_eye_yaw: 0.0,
            right_eye_pitch: 0.0,
            right_eye_yaw: 0.0,
            routine_left_lid: self.phase.lid(),
            routine_right_lid: self.phase.lid(),
            routine_brow_raise: 0.0,
            routine_brow_angry: 0.0,
            routine_widen: 0.0,
            routine_squint: 0.0,
            routine_dilate: 0.0,
            timestamp: time,
            video_timestamp_left: time,
            video_timestamp_right: time,
            routine_state: RoutineState::from_raw(self.phase.flags()),
            jpeg_left_len: 0,
            jpeg_right_len: 0,
        };
        self.push_frame(meta, left, right);
    }

    fn push_frame(&mut self, meta: FrameMeta, left: &Frame, right: &Frame) {
        let jpeg_left = encode_jpeg(&left.image);
        let jpeg_right = encode_jpeg(&right.image);

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
