//! Per-eye supervision labels.

/// Expression label, channel order `[lid, widen, squint, brow]`.
#[derive(Copy, Clone, Debug)]
pub struct Expr {
    pub lid: f32,
    pub widen: f32,
    pub squint: f32,
    pub brow: f32,
}

impl Expr {
    pub fn to_array(self) -> [f32; 4] {
        [self.lid, self.widen, self.squint, self.brow]
    }

    pub fn from_array([lid, widen, squint, brow]: [f32; 4]) -> Self {
        Self { lid, widen, squint, brow }
    }
}

/// Gaze label, channel order `[pitch, yaw]`.
#[derive(Copy, Clone, Debug)]
pub struct Gaze {
    pub pitch: f32,
    pub yaw: f32,
}

impl Gaze {
    pub fn to_array(self) -> [f32; 2] {
        [self.pitch, self.yaw]
    }
}
