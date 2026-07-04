use std::{borrow::Cow, cell::Cell, thread, thread::sleep, time::Duration};

use indicatif::MultiProgress;
use snout::{
    calibration::EyeShape, capture::discovery::{query_cameras, CameraInfo}, config::Config, track::{eye::EyeTracker, face::FaceTracker, initialize_runtime, output::Output},
};

use crate::status::{Heartbeat, Pair, Rate, StatusBar, StatusBarItem};

const IDLE_RETRY: Duration = Duration::from_millis(10);

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
        let cameras = &cameras;

        thread::scope(|scope| {
            scope.spawn(move || self.run_face(cameras, multi));
            scope.spawn(move || self.run_eye(cameras, multi));
        });
    }

    /// Face tracking worker: owns its tracker, output, and status line.
    fn run_face(&self, cameras: &[CameraInfo], multi: &MultiProgress) {
        let mut tracker = FaceTracker::with_config(cameras, &self.config).unwrap();
        let mut output = Output::with_config(&self.config).unwrap();

        let mut status = StatusBar::new(multi);
        let heartbeat = status.add(Heartbeat::new("FACE", Duration::from_secs(1)));
        let tick_rate = status.add(Rate::new("TICK", 0));

        loop {
            match tracker.track().unwrap() {
                Some(report) => {
                    heartbeat.beat();
                    output.send_face(report.weights);
                    output.flush().unwrap();
                    tick_rate.inc();
                    self.throttle();
                }
                None => sleep(IDLE_RETRY),
            }

            status.display();
        }
    }

    /// Eye tracking worker: owns its tracker, output, and status line.
    fn run_eye(&self, cameras: &[CameraInfo], multi: &MultiProgress) {
        let mut tracker = EyeTracker::with_config(cameras, &self.config).unwrap();
        let mut output = Output::with_config(&self.config).unwrap();

        let mut status = StatusBar::new(multi);
        let heartbeat = status.add(Heartbeat::new("EYE", Duration::from_secs(1)));
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
            match tracker.track().unwrap() {
                Some(report) => {
                    heartbeat.beat();

                    if let Some((left, right, lids, brow, widen, squint)) = &eye_debug {
                        left.set(
                            report.weights.get(EyeShape::LeftEyePitch).unwrap_or(0.),
                            report.weights.get(EyeShape::LeftEyeYaw).unwrap_or(0.),
                        );
                        right.set(
                            report.weights.get(EyeShape::RightEyePitch).unwrap_or(0.),
                            report.weights.get(EyeShape::RightEyeYaw).unwrap_or(0.),
                        );
                        lids.set(
                            report.weights.get(EyeShape::LeftEyeLid).unwrap_or(0.),
                            report.weights.get(EyeShape::RightEyeLid).unwrap_or(0.),
                        );
                        brow.set(
                            report.weights.get(EyeShape::LeftEyeBrow).unwrap_or(0.),
                            report.weights.get(EyeShape::RightEyeBrow).unwrap_or(0.),
                        );
                        widen.set(
                            report.weights.get(EyeShape::LeftEyeWiden).unwrap_or(0.),
                            report.weights.get(EyeShape::RightEyeWiden).unwrap_or(0.),
                        );
                        squint.set(
                            report.weights.get(EyeShape::LeftEyeSquint).unwrap_or(0.),
                            report.weights.get(EyeShape::RightEyeSquint).unwrap_or(0.),
                        );
                    }

                    output.send_eyes(report.weights);
                    output.flush().unwrap();
                    tick_rate.inc();
                    self.throttle();
                }
                None => sleep(IDLE_RETRY),
            }

            status.display();
        }
    }

    /// Per-tick throttle, matching the single-loop behavior: sleep for the
    /// configured `interval` (skipped when it's 0), or 10ms by default.
    fn throttle(&self) {
        if let Some(interval) = self.config.interval {
            if interval > 0 {
                sleep(Duration::from_millis(interval));
            }
        } else {
            sleep(Duration::from_millis(10));
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
