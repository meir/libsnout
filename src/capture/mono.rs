use crate::capture::{
    CameraError, Frame,
    discovery::CameraSource,
    internal::{Camera, V4lCamera, http::HttpCamera},
    sensor::SensorConfig,
};

pub struct MonoCamera {
    inner: Box<dyn Camera>,
    frame: Option<Frame>,
}

impl MonoCamera {
    pub fn open(source: &CameraSource) -> Result<Self, CameraError> {
        let inner: Box<dyn Camera> = match source {
            CameraSource::V4l(s) => Box::new(V4lCamera::open(*s)?),
            CameraSource::Http(s) => Box::new(HttpCamera::open(s)?),
        };

        Ok(Self { frame: None, inner })
    }

    pub fn get_frame(&mut self) -> Result<&Frame, CameraError> {
        let image = self.inner.read_frame()?;
        self.frame = Some(Frame::new(image));

        self.frame
            .as_ref()
            .ok_or_else(|| CameraError::Internal("No frame available".to_string()))
    }

    /// Records a manual sensor configuration, applied to the underlying
    /// hardware once the stream is live (see [`Camera::set_sensor_config`]).
    pub fn set_sensor_config(&mut self, config: SensorConfig) {
        self.inner.set_sensor_config(config);
    }
}
