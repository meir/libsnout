mod commands;

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use crate::commands::{
    CaptureCommand, ListCamerasCommand, SampleCommand, TrackCommand, TrainCommand,
};

fn main() {
    let cli = Args::parse();

    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
            )
            .init();
    }

    let config_path = cli
        .config
        .or_else(snout::config::find_default_config)
        .unwrap_or_else(|| {
            eprintln!("Error: No config file found.");
            eprintln!("Specify a config file with --config <path>, or place one in a standard location.");
            process::exit(1);
        });

    let config = snout::config::load(&config_path).unwrap();

    match cli.command {
        Commands::ListCameras {} => ListCamerasCommand::new().run(),
        Commands::Train {
            source,
            destination,
        } => TrainCommand::new(source, destination).run(),
        Commands::Track { eye_debug } => TrackCommand::new(config, eye_debug).run(),
        Commands::Capture {
            source,
            destination,
        } => CaptureCommand::new(config, source, destination).run(),
        Commands::Sample { output, stage } => {
            let stages: Vec<snout::sample::sampler::Stage> =
                stage.into_iter().map(Into::into).collect();
            SampleCommand::new(config, output, stages).run()
        }
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(flatten_help = true)]
struct Args {
    #[arg(short, long, value_name = "config.toml")]
    config: Option<PathBuf>,

    /// Enable verbose output (tracing logs).
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CaptureSource {
    LeftEye,
    RightEye,
    Face,
}

/// A single calibration pass, for `sample --stage`.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum Stage {
    Gaze,
    FreeExpr,
    Blink,
    Widen,
    Squint,
    Brow,
}

impl From<Stage> for snout::sample::sampler::Stage {
    fn from(stage: Stage) -> Self {
        use snout::sample::sampler::Stage as S;
        match stage {
            Stage::Gaze => S::Gaze,
            Stage::FreeExpr => S::FreeExpr,
            Stage::Blink => S::Blink,
            Stage::Widen => S::Widen,
            Stage::Squint => S::Squint,
            Stage::Brow => S::Brow,
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// List all available cameras.
    ListCameras {},
    /// Save a frame from the specified source.
    Capture {
        source: CaptureSource,
        #[arg(value_name = "output.jpeg")]
        destination: PathBuf,
    },
    /// Train the MobileNetV4 dual-eye model (gaze + expression) and export ONNX.
    Train {
        /// A capture `.bin` file or a session directory (e.g. from `sample`).
        #[arg(value_name = "capture")]
        source: PathBuf,
        /// A destination `onnx` file.
        #[arg(value_name = "output.onnx")]
        destination: PathBuf,
    },
    /// Start tracking!
    Track {
        #[arg(long)]
        eye_debug: bool,
    },
    /// Run calibration sampling into a session directory.
    Sample {
        /// Output session directory.
        #[arg(short, long, default_value = "my_session")]
        output: PathBuf,
        /// Record only these stage(s); omit to record the full session.
        #[arg(short, long, value_enum)]
        stage: Vec<Stage>,
    },
}
