mod track;

use std::{io::Write, path::PathBuf};

use snout::{
    capture::discovery::query_cameras,
    config::Config,
    sample::sampler::{LongSampler, Stage},
    track::{eye::EyeTracker, face::FaceTracker, initialize_runtime},
};

pub use track::TrackCommand;

use crate::CaptureSource;

pub struct ListCamerasCommand {}

impl ListCamerasCommand {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self) {
        let cameras = snout::capture::discovery::query_cameras();
        for camera in cameras {
            println!("{}", camera.display_name());
        }
    }
}

/// Trains the MobileNetV4-based dual-eye model (gaze + expression) and exports ONNX.
pub struct TrainCommand {
    source: PathBuf,
    destination: PathBuf,
}

impl TrainCommand {
    pub fn new(source: PathBuf, destination: PathBuf) -> Self {
        Self {
            source,
            destination,
        }
    }

    pub fn run(&self) {
        use snout_train::train::Trainer;
        use snout_train::{TrainBackend, default_device};

        println!("Training dual-eye model...");
        let device = default_device();
        let result = Trainer::<TrainBackend>::new(device)
            .on_progress(print_progress)
            .train_to_onnx(&self.source, &self.destination);

        match result {
            Ok(()) => {
                println!();
                println!("wrote: {}", self.destination.display());
                println!("training completed successfully.");
            }
            Err(e) => {
                eprintln!("\ntraining failed: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn print_progress(p: snout_train::train::Progress) {
    if let Some(val) = p.val {
        println!(
            "\r{:<4}  step {:>5}/{:<5}  loss {:.5}   val_mse {:.5}  r[lid={:.2} widen={:.2} squint={:.2} brow={:.2}]",
            p.phase, p.step, p.total_steps, p.loss,
            val.mse, val.corr[0], val.corr[1], val.corr[2], val.corr[3],
        );
    } else {
        print!(
            "\r{:<4}  step {:>5}/{:<5}  loss {:.5}    ",
            p.phase, p.step, p.total_steps, p.loss,
        );
        let _ = std::io::stdout().flush();
    }
}

pub struct CaptureCommand {
    config: Config,
    source: CaptureSource,
    destination: PathBuf,
}

impl CaptureCommand {
    pub fn new(config: Config, source: CaptureSource, destination: PathBuf) -> Self {
        Self {
            config,
            source,
            destination,
        }
    }

    pub fn run(&self) {
        let cameras = query_cameras();

        initialize_runtime(self.config.libonnxruntime.as_ref());

        {
            match self.source {
                CaptureSource::LeftEye => {
                    let mut tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.left_processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }
                        i += 1;
                    }
                    println!("Could not capture frame");
                }
                CaptureSource::RightEye => {
                    let mut tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.right_processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }

                        i += 1;
                    }
                    println!("Could not capture frame");
                }
                CaptureSource::Face => {
                    let mut tracker = FaceTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }

                        i += 1;
                    }
                    println!("Could not capture frame");
                }
            }
        }
    }
}

pub struct SampleCommand {
    config: Config,
    output: PathBuf,
    stages: Vec<Stage>,
}

impl SampleCommand {
    pub fn new(config: Config, output: PathBuf, stages: Vec<Stage>) -> Self {
        Self {
            config,
            output,
            stages,
        }
    }

    pub fn run(&self) {
        let cameras = query_cameras();

        let mut sampler = LongSampler::with_config(&cameras, &self.config)
            .expect("failed to create sampler");

        if self.stages.is_empty() {
            println!("Starting full calibration...");
        } else {
            let names: Vec<_> = self.stages.iter().map(|s| s.file_name()).collect();
            println!("Recording stage(s): {}", names.join(", "));
        }

        sampler.run(&self.output, &self.stages).expect("sampling failed");
        println!("Done. Written to {}", self.output.display());
    }
}
