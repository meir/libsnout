pub mod http;
pub mod v4l;

use image::GrayImage;
pub use v4l::V4lCamera;

use crate::capture::CameraError;
use crate::capture::sensor::SensorConfig;

pub trait Camera {
    fn read_frame(&mut self) -> Result<GrayImage, CameraError>;

    /// Records a manual sensor configuration to be applied to the underlying
    /// hardware once the stream is live. Cameras that don't support direct
    /// sensor control (or don't match the config's hardware) ignore it.
    fn set_sensor_config(&mut self, _config: SensorConfig) {}
}
