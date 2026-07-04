//! Manual sensor controls applied on top of a camera stream.
//!
//! Some boards ignore the standard V4L2/UVC exposure controls and hide
//! everything behind an on-sensor auto-exposure loop. For those we talk to the
//! sensor directly (see the [`gc0308`] crate). Each supported sensor is a
//! variant of [`SensorConfig`]; the values only take effect on hardware that
//! actually matches, and are an inert no-op everywhere else.

use serde::{Deserialize, Serialize};

/// A manual configuration for a specific image sensor.
///
/// The active variant is matched against the hardware behind the camera at
/// apply time, so an unrelated camera silently ignores it.
#[derive(Clone, Debug)]
pub enum SensorConfig {
    Gc0308(Gc0308Config),
}

/// Manual controls for the GC0308 sensor (over the Sonix SN9C29x bridge).
///
/// Each field is optional; omitted values fall back to the sensor defaults
/// applied when its auto-exposure loop is disabled.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Gc0308Config {
    /// Fixed exposure, `0..=4095`.
    pub exposure: u16,
    /// Fixed global gain, `0..=255`.
    pub gain: u8,
}
