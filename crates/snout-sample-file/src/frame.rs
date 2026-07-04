use crate::flags::RoutineState;

pub const FRAME_META_SIZE: usize = 100;
pub const MAX_JPEG_SIZE: usize = 10 * 1024 * 1024;

/// Layout: `=ffffffffffffffffqqqiii` (little-endian, no padding)
/// - 16 × f32 (64 bytes)
/// - 3 × i64 (24 bytes)
/// - 3 × i32 (12 bytes)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeta {
    pub routine_pitch: f32,
    pub routine_yaw: f32,
    pub routine_distance: f32,
    pub routine_convergence: f32,
    pub fov_adjust_distance: f32,

    pub left_eye_pitch: f32,
    pub left_eye_yaw: f32,
    pub right_eye_pitch: f32,
    pub right_eye_yaw: f32,

    pub routine_left_lid: f32,
    pub routine_right_lid: f32,

    pub routine_brow_raise: f32,
    pub routine_brow_angry: f32,
    pub routine_widen: f32,
    pub routine_squint: f32,
    pub routine_dilate: f32,

    pub timestamp: i64,
    pub video_timestamp_left: i64,
    pub video_timestamp_right: i64,

    pub routine_state: RoutineState,
    pub jpeg_left_len: i32,
    pub jpeg_right_len: i32,
}

impl FrameMeta {
    pub fn is_good_data(&self) -> bool {
        self.routine_state.is_good_data()
    }

    pub fn is_gaze_data(&self) -> bool {
        self.routine_state.is_gaze_data()
    }

    pub fn is_expr_unlabeled(&self) -> bool {
        self.routine_state.is_expr_unlabeled()
    }
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub meta: FrameMeta,
    pub jpeg_left: Vec<u8>,
    pub jpeg_right: Vec<u8>,
}
