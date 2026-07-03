use std::{borrow::Cow, cell::Cell, thread::sleep, time::Duration};

use indicatif::MultiProgress;
use snout::{
    calibration::EyeShape, capture::discovery::query_cameras, config::Config, track::{eye::EyeTracker, face::FaceTracker, initialize_runtime, output::Output},
};

use crate::status::{Heartbeat, Pair, Rate, StatusBar, StatusBarItem};

pub struct TrackCommand {
    config: Config,
    eye_debug: bool,
}

impl TrackCommand {
    pub fn new(config: Config, eye_debug: bool) -> Self {
        Self { config, eye_debug }
    }

    pub fn run(&self, multi: &MultiProgress) {
        initialize_runtime(self.config.libonnxruntime.as_ref());

        let cameras = query_cameras();

        let mut face_tracker = FaceTracker::with_config(&cameras, &self.config).unwrap();
        let mut eye_tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

        let mut output = Output::with_config(&self.config).unwrap();

        let mut status = StatusBar::new(multi);
        let face_heartbeat = status.add(Heartbeat::new("FACE", Duration::from_secs(1)));
        let eye_heartbeat = status.add(Heartbeat::new("EYE", Duration::from_secs(1)));
        let tick_rate = status.add(Rate::new("TICK", 0));

        let eye_debug = self.eye_debug.then(|| {
            let left = status.add(Gaze::new("L"));
            let right = status.add(Gaze::new("R"));
            let lids = status.add(Pair::new("LIDS"));
            let brow = status.add(Pair::new("BROW"));
            let widen = status.add(Pair::new("WIDEN"));
            let squint = status.add(Pair::new("SQUINT"));
            (left, right, lids, brow, widen, squint)
        });

        loop {
            let face_report = face_tracker.track().unwrap();
            let eye_report = eye_tracker.track().unwrap();

            if let Some(face_report) = face_report {
                face_heartbeat.beat();
                output.send_face(face_report.weights);
            }

            if let Some(eye_report) = eye_report {
                eye_heartbeat.beat();

                if let Some((left, right, lids, brow, widen, squint)) = &eye_debug {
                    left.set(
                        eye_report.weights.get(EyeShape::LeftEyePitch).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::LeftEyeYaw).unwrap_or(0.),
                    );
                    right.set(
                        eye_report.weights.get(EyeShape::RightEyePitch).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::RightEyeYaw).unwrap_or(0.),
                    );
                    lids.set(
                        eye_report.weights.get(EyeShape::LeftEyeLid).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::RightEyeLid).unwrap_or(0.),
                    );
                    brow.set(
                        eye_report.weights.get(EyeShape::LeftEyeBrow).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::RightEyeBrow).unwrap_or(0.),
                    );
                    widen.set(
                        eye_report.weights.get(EyeShape::LeftEyeWiden).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::RightEyeWiden).unwrap_or(0.),
                    );
                    squint.set(
                        eye_report.weights.get(EyeShape::LeftEyeSquint).unwrap_or(0.),
                        eye_report.weights.get(EyeShape::RightEyeSquint).unwrap_or(0.),
                    );
                }

                output.send_eyes(eye_report.weights);
            }

            output.flush().unwrap();

            tick_rate.inc();
            status.display();

            if let Some(interval) = self.config.interval {
                if interval > 0 {
                    sleep(Duration::from_millis(interval));
                }
            } else {
                sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Per-eye gaze readout, rendered as `L(+0.12,-0.34)`.
struct Gaze {
    side: &'static str,
    pitch: Cell<f32>,
    yaw: Cell<f32>,
}

impl Gaze {
    fn new(side: &'static str) -> Self {
        Self {
            side,
            pitch: Cell::new(0.0),
            yaw: Cell::new(0.0),
        }
    }

    fn set(&self, pitch: f32, yaw: f32) {
        self.pitch.set(pitch);
        self.yaw.set(yaw);
    }
}

impl StatusBarItem for Gaze {
    fn render(&self) -> Cow<'static, str> {
        format!("{}({:+.2},{:+.2})", self.side, self.pitch.get(), self.yaw.get()).into()
    }
}
