use std::io;
use std::path::Path;
use std::time::Duration;

use crate::capture::{
    CameraError, StereoCamera,
    discovery::{CameraInfo, CameraSource},
    processing::FramePreprocessor,
};
use crate::config::{Config, OverlayMode};
use crate::sample::collector::{FrameCollector, Phase};
use crate::sample::net::{Event, Mode, Overlay, Routine};

#[derive(Debug, thiserror::Error)]
pub enum SamplerError {
    #[error("no sample config in config file")]
    NoConfig,
    #[error("overlay error: {0}")]
    Overlay(#[from] io::Error),
    #[error("camera error: {0}")]
    Camera(String),
}

/// One calibration pass. Each stage is a guided tutorial followed by the recorded
/// routine, written to its own `<stage>.bin` in the session directory.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stage {
    Gaze,
    FreeExpr,
    Blink,
    Widen,
    Squint,
    Brow,
}

impl Stage {
    /// Canonical record order — must match the reader's session order
    /// (`snout-train` `data/capture.rs::SESSION_STAGES`).
    pub const ALL: [Stage; 6] = [
        Stage::Gaze,
        Stage::FreeExpr,
        Stage::Blink,
        Stage::Widen,
        Stage::Squint,
        Stage::Brow,
    ];

    /// The bin file stem for this stage (`<file_name>.bin`).
    pub fn file_name(self) -> &'static str {
        match self {
            Stage::Gaze => "gaze",
            Stage::FreeExpr => "free-expr",
            Stage::Blink => "blink",
            Stage::Widen => "widen",
            Stage::Squint => "squint",
            Stage::Brow => "brow",
        }
    }

    fn tutorial(self) -> Routine {
        match self {
            Stage::Gaze => Routine::GazeTutorial,
            Stage::FreeExpr => Routine::GazeExprTutorial,
            Stage::Blink => Routine::BlinkTutorial,
            Stage::Widen => Routine::WidenTutorial,
            Stage::Squint => Routine::SquintTutorial,
            Stage::Brow => Routine::BrowTutorial,
        }
    }

    fn routine(self) -> Routine {
        match self {
            Stage::Gaze => Routine::Gaze(Duration::from_secs(60)),
            Stage::FreeExpr => Routine::FreeExpr(Duration::from_secs(60)),
            Stage::Blink => Routine::Blink(Duration::from_secs(10)),
            Stage::Widen => Routine::Widen(Duration::from_secs(20)),
            Stage::Squint => Routine::Squint(Duration::from_secs(20)),
            Stage::Brow => Routine::Brow(Duration::from_secs(20)),
        }
    }
}

pub struct LongSampler {
    overlay_path: String,
    overlay_mode: Mode,
    left_preprocessor: FramePreprocessor,
    right_preprocessor: FramePreprocessor,
    camera: Option<StereoCamera>,
    left_source: Option<CameraSource>,
    right_source: Option<CameraSource>,
}

impl LongSampler {
    pub fn with_config(cameras: &[CameraInfo], config: &Config) -> Result<Self, SamplerError> {
        let sample_config = config.sample.as_ref().ok_or(SamplerError::NoConfig)?;

        let overlay_mode = match sample_config.overlay.mode {
            OverlayMode::OpenVr => Mode::OpenVr,
            OverlayMode::OpenXr => Mode::OpenXr,
            OverlayMode::Debug => Mode::Debug,
        };

        let left_source = cameras
            .iter()
            .find(|s| s.display_name() == config.eye.left.camera)
            .map(|c| c.source.clone());

        let right_source = cameras
            .iter()
            .find(|s| s.display_name() == config.eye.right.camera)
            .map(|c| c.source.clone());

        let mut left_preprocessor = FramePreprocessor::new();
        left_preprocessor.set_crop(config.eye.left.crop);
        if let Some(transform) = &config.eye.left.transform {
            left_preprocessor.set_config(*transform);
        }

        let mut right_preprocessor = FramePreprocessor::new();
        right_preprocessor.set_crop(config.eye.right.crop);
        if let Some(transform) = &config.eye.right.transform {
            right_preprocessor.set_config(*transform);
        }

        Ok(Self {
            overlay_path: sample_config.overlay.path.to_string_lossy().into_owned(),
            overlay_mode,
            left_preprocessor,
            right_preprocessor,
            camera: None,
            left_source,
            right_source,
        })
    }

    /// Records calibration into the session directory `dir` as one bin per stage.
    ///
    /// With an empty `stages` slice the full session is recorded (all of [`Stage::ALL`]);
    /// otherwise only the listed stages are (re-)recorded, overwriting just their bins.
    pub fn run(&mut self, dir: impl AsRef<Path>, stages: &[Stage]) -> Result<(), SamplerError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        self.ensure_camera()?;

        let mut overlay = Overlay::start(&self.overlay_path, self.overlay_mode)?;

        let stages = if stages.is_empty() { &Stage::ALL[..] } else { stages };
        for &stage in stages {
            self.record_stage(&mut overlay, dir, stage)?;
        }

        self.finish(&mut overlay)
    }

    /// Runs one stage's tutorial + recording and writes `<dir>/<stage>.bin`.
    /// Each stage gets a fresh collector, so a crash leaves completed stages on disk.
    fn record_stage(
        &mut self,
        overlay: &mut Overlay,
        dir: &Path,
        stage: Stage,
    ) -> Result<(), SamplerError> {
        let mut collector = FrameCollector::new();
        self.routine(overlay, &mut collector, stage.tutorial())?;
        self.routine(overlay, &mut collector, stage.routine())?;

        let path = dir.join(format!("{}.bin", stage.file_name()));
        backup_existing(&path)?;
        collector.write(&path)?;
        Ok(())
    }

    fn finish(&mut self, overlay: &mut Overlay) -> Result<(), SamplerError> {
        // Switch to *any* other routine to trigger sound
        overlay.begin(Routine::Trainer)?;
        std::thread::sleep(Duration::from_secs(1));

        overlay.close()?;

        Ok(())
    }

    fn routine(&mut self, overlay: &mut Overlay, collector: &mut FrameCollector, routine: Routine) -> Result<(), SamplerError> {
        overlay.begin(routine)?;

        if routine.is_tutorial() {
            self.wait_for_finish(overlay)?;
        } else {
            let phase = Phase::from_routine(routine);
            self.collect(overlay, collector, phase)?;
        }

        Ok(())
    }

    fn collect(
        &mut self,
        overlay: &mut Overlay,
        collector: &mut FrameCollector,
        phase: Phase,
    ) -> Result<(), SamplerError> {
        loop {
            match overlay.try_recv()? {
                Some(Event::Position(pos)) => {
                    collector.set_position(pos);
                }
                Some(Event::Finished) => return Ok(()),
                None => {}
            }

            if let Some((left, right)) = self.grab_frame()? {
                collector.add(phase, &left, &right);
            }
        }
    }

    fn wait_for_finish(&mut self, overlay: &mut Overlay) -> Result<(), SamplerError> {
        loop {
            match overlay.try_recv()? {
                Some(Event::Finished) => return Ok(()),
                _ => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    }

    fn grab_frame(
        &mut self,
    ) -> Result<Option<(crate::capture::Frame, crate::capture::Frame)>, SamplerError> {
        let camera = self.camera.as_mut().unwrap();

        let (left_raw, right_raw) = match camera.get_frames() {
            Ok(frames) => frames,
            Err(CameraError::InvalidFrame(_)) => return Ok(None),
            Err(e) => return Err(SamplerError::Camera(e.to_string())),
        };

        let left = self
            .left_preprocessor
            .process(left_raw)
            .map_err(|e| SamplerError::Camera(e.to_string()))?
            .clone();

        let right = self
            .right_preprocessor
            .process(right_raw)
            .map_err(|e| SamplerError::Camera(e.to_string()))?
            .clone();

        Ok(Some((left, right)))
    }

    fn ensure_camera(&mut self) -> Result<(), SamplerError> {
        if self.camera.is_none() {
            let (Some(left), Some(right)) = (&self.left_source, &self.right_source) else {
                return Err(SamplerError::Camera("no camera source configured".into()));
            };

            let camera = if left == right {
                StereoCamera::open_sbs(left)
            } else {
                StereoCamera::open(left, right)
            }
            .map_err(|e| SamplerError::Camera(e.to_string()))?;

            self.camera = Some(camera);
        }

        Ok(())
    }
}

/// Moves `path` into a sibling `backups/` directory with a timestamped name if it
/// already exists, so re-recording a stage never destroys the previous pass. The
/// reader (`read_capture`) only looks for `<stage>.bin` in the session root, so backups
/// are ignored by training; restore one by copying it back over `<stage>.bin`.
fn backup_existing(path: &Path) -> Result<(), SamplerError> {
    if !path.exists() {
        return Ok(());
    }

    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("stage");

    let backups = dir.join("backups");
    std::fs::create_dir_all(&backups)?;

    // Timestamped name; bump a counter on the (real-world impossible, but cheap to guard)
    // chance two backups land in the same millisecond, so we never clobber a backup.
    let ts = backup_timestamp();
    let mut dest = backups.join(format!("{stem}-{ts}.bin"));
    let mut n = 1;
    while dest.exists() {
        dest = backups.join(format!("{stem}-{ts}-{n}.bin"));
        n += 1;
    }
    std::fs::rename(path, dest)?;
    Ok(())
}

/// Milliseconds since the Unix epoch, for unique, chronologically-sortable backup names.
fn backup_timestamp() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{backup_existing, backup_timestamp};
    use std::fs;

    #[test]
    fn backup_preserves_previous_passes() {
        let dir = std::env::temp_dir()
            .join(format!("snout_backup_{}_{}", std::process::id(), backup_timestamp()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("squint.bin");
        let backups = dir.join("backups");

        // nothing to back up yet -> no-op, no backups dir created.
        backup_existing(&path).unwrap();
        assert!(!backups.exists());

        // first re-record: the existing bin is moved aside.
        fs::write(&path, b"pass-1").unwrap();
        backup_existing(&path).unwrap();
        assert!(!path.exists(), "current bin should be moved out");
        assert_eq!(fs::read_dir(&backups).unwrap().count(), 1);

        // second re-record: previous pass is *also* kept (not clobbered).
        fs::write(&path, b"pass-2").unwrap();
        backup_existing(&path).unwrap();
        assert_eq!(fs::read_dir(&backups).unwrap().count(), 2);

        fs::remove_dir_all(&dir).ok();
    }
}
