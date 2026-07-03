use crate::{
    calibration::{FaceShape, ManualFaceCalibrator},
    capture::{
        CameraError, Frame, MonoCamera,
        discovery::{CameraInfo, CameraSource, resolve_source},
        processing::FramePreprocessor,
        sensor::SensorConfig,
    },
    config::Config,
    pipeline::FacePipeline,
    track::TrackerError,
    weights::Weights,
};

pub struct FaceReport<'a> {
    pub raw_frame: &'a Frame,
    pub processed_frame: &'a Frame,
    pub weights: &'a Weights<FaceShape>,
}

pub struct FaceTracker {
    pub preprocessor: FramePreprocessor,
    pub pipeline: FacePipeline,
    pub calibrator: ManualFaceCalibrator,

    camera: Option<MonoCamera>,
    source: Option<CameraSource>,
    sensor_config: Option<SensorConfig>,
}

impl FaceTracker {
    pub fn new() -> Self {
        Self {
            preprocessor: FramePreprocessor::new(),
            pipeline: FacePipeline::new(),
            calibrator: ManualFaceCalibrator::new(),

            camera: None,
            source: None,
            sensor_config: None,
        }
    }

    pub fn with_config(cameras: &[CameraInfo], config: &Config) -> Result<Self, TrackerError> {
        let mut tracker = Self::new();

        tracker.pipeline.set_model(config.face.model.as_ref())?;

        if let Some(filter) = config.face.filter {
            tracker.pipeline.set_filter(filter);
        }

        let camera = resolve_source(cameras, &config.face.camera);

        tracker.set_source(camera);

        if let Some(gc0308) = config.face.gc0308.clone() {
            tracker.sensor_config = Some(SensorConfig::Gc0308(gc0308));
        }

        tracker.preprocessor.set_crop(config.face.crop);

        if let Some(transform) = &config.face.transform {
            tracker.preprocessor.set_config(*transform);
        }

        for calibration in &config.face.calibration {
            tracker.calibrator.set_upper(calibration.shape, calibration.upper);
            tracker.calibrator.set_lower(calibration.shape, calibration.lower);
        }

        Ok(tracker)
    }

    /// Sets the camera source for the tracker.
    ///
    /// If the source is different from the current source, the camera is reset.
    pub fn set_source(&mut self, source: Option<CameraSource>) {
        if self.source != source {
            self.camera = None;
        }

        self.source = source;
    }

    pub fn track(&mut self) -> Result<Option<FaceReport<'_>>, TrackerError> {
        if !self.ensure_camera()? {
            return Ok(None);
        }

        let camera = self.camera.as_mut().unwrap();

        let raw_frame = match camera.get_frame() {
            Ok(frame) => frame,
            Err(CameraError::InvalidFrame(_)) => {
                // TODO: Keep track of the amount of invalid frames
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };

        let processed_frame = self.preprocessor.process(raw_frame)?;

        let Some(raw_weights) = self.pipeline.run(processed_frame)? else {
            return Ok(None);
        };

        let weights = self.calibrator.calibrate(raw_weights);

        Ok(Some(FaceReport {
            raw_frame,
            processed_frame,
            weights,
        }))
    }

    fn ensure_camera(&mut self) -> Result<bool, TrackerError> {
        if self.camera.is_none() {
            let Some(source) = &self.source else {
                return Ok(false);
            };

            let mut camera =
                MonoCamera::open(source).map_err(|e| TrackerError::Open(e.to_string()))?;

            if let Some(sensor_config) = &self.sensor_config {
                camera.set_sensor_config(sensor_config.clone());
            }

            self.camera = Some(camera);
        }

        Ok(true)
    }
}
