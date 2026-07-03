use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{calibration::FaceShape, capture::{processing::{Crop, PreprocessConfig}, sensor::Gc0308Config}, pipeline::FilterParameters};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("No file found.")]
    FileNotFound,
    #[error("Invalid format: {0}")]
    InvalidConfig(String),
}

fn resolve_path(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_relative() {
        base.join(path)
    } else {
        path
    }
}

/// Finds a config file in pre-set locations.
///
/// Checks the following locations:
/// - `$XDG_CONFIG_HOME/snout/config.toml`
/// - `$HOME/.config/snout/config.toml`
/// - `$HOME/.snout/config.toml`
/// - `/etc/snout/config.toml`
pub fn find_default_config() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        let path = PathBuf::from(xdg).join("snout/config.toml");
        if path.exists() {
            return Some(path);
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let path = PathBuf::from(&home).join(".config/snout/config.toml");
        if path.exists() {
            return Some(path);
        }

        let path = PathBuf::from(&home).join(".snout/config.toml");
        if path.exists() {
            return Some(path);
        }
    }

    let path = PathBuf::from("/etc/snout/config.toml");
    if path.exists() {
        return Some(path);
    }

    None
}

pub fn load(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
    let path = path.as_ref();
    let base = path.parent().unwrap_or(Path::new("."));

    let str = std::fs::read_to_string(path).map_err(|_| ConfigError::FileNotFound)?;
    let mut config: Config =
        toml::from_str(&str).map_err(|e| ConfigError::InvalidConfig(e.to_string()))?;

    config.libonnxruntime = config.libonnxruntime.map(|p| resolve_path(base, p));
    config.eye.model = config.eye.model.map(|p| resolve_path(base, p));
    config.face.model = config.face.model.map(|p| resolve_path(base, p));

    Ok(config)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceShapeCalibration {
    pub shape: FaceShape,
    pub lower: f32,
    pub upper: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub libonnxruntime: Option<PathBuf>,

    pub eye: EyesConfig,
    pub face: FaceConfig,

    #[serde(default)]
    pub sample: Option<SampleConfig>,

    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EyesConfig {
    pub link: Option<bool>,
    pub model: Option<PathBuf>,
    pub filter: Option<FilterParameters>,

    pub left: EyeConfig,
    pub right: EyeConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EyeConfig {
    pub camera: String,
    #[serde(default)]
    pub crop: Crop,
    pub transform: Option<PreprocessConfig>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceConfig {
    pub camera: String,
    pub model: Option<PathBuf>,
    pub filter: Option<FilterParameters>,
    #[serde(default)]
    pub crop: Crop,
    pub transform: Option<PreprocessConfig>,

    #[serde(default)]
    pub calibration: Vec<FaceShapeCalibration>,

    /// Manual GC0308 sensor controls, applied only if the face camera is a
    /// GC0308 board. Left untouched when absent.
    pub gc0308: Option<Gc0308Config>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SampleConfig {
    pub overlay: OverlayConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OverlayConfig {
    pub path: PathBuf,
    pub mode: OverlayMode,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "PascalCase")]
pub enum OverlayMode {
    OpenVr,
    OpenXr,
    Debug,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub osc: OscConfig,

    pub vrchat: Option<VrchatOscConfig>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OscConfig {
    pub destination: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VrchatOscConfig {
    pub destination: String,

    #[serde(default = "VrchatOscConfig::default_max_yaw")]
    pub max_yaw: f32,

    #[serde(default = "VrchatOscConfig::default_max_pitch")]
    pub max_pitch: f32,
}

impl VrchatOscConfig {
    fn default_max_yaw() -> f32 {
        30.0
    }

    fn default_max_pitch() -> f32 {
        20.0
    }
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            destination: "127.0.0.1:9400".to_string(),
        }
    }
}
