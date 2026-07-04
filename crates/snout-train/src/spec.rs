//! The fixed shape of the eye-tracking problem.
//!
//! These domain constants are shared by every pillar (model, data, train, export),
//! so they live at the crate root with no dependencies of their own.

/// Per-eye frame size the capture pipeline emits (grayscale, `[0, 1]`).
pub const IMAGE_WIDTH: usize = 128;
/// Per-eye frame height (see [`IMAGE_WIDTH`]).
pub const IMAGE_HEIGHT: usize = 128;
/// Pixels in a single frame.
pub const PIXELS_PER_FRAME: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// Temporal stack depth: the current frame plus three predecessors of the same eye.
/// The stack is fed to each tower as its input channels.
pub const TEMPORAL_DEPTH: usize = 4;
/// Input channels per tower (one per frame in the temporal stack).
pub const PER_EYE_CHANNELS: usize = TEMPORAL_DEPTH;

/// Gaze outputs per eye: `[pitch, yaw]`.
pub const GAZE_OUTPUTS: usize = 2;
/// Expression outputs per eye: `[lid, widen, squint, brow]`.
pub const EXPR_OUTPUTS: usize = 4;
/// Outputs of one per-eye tower: gaze followed by expression.
pub const PER_EYE_OUTPUTS: usize = GAZE_OUTPUTS + EXPR_OUTPUTS;
/// Outputs of the merged deployment model: both eyes.
pub const MERGED_OUTPUTS: usize = 2 * PER_EYE_OUTPUTS;
