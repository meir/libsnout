#![allow(dead_code)] // Since we have some unused packets.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Packet {
    #[serde(rename = "PacketName")]
    pub name: String,
    #[serde(rename = "PacketData")]
    pub data: serde_json::Value,
}

impl Packet {
    pub fn new<T: Serialize>(name: &str, data: &T) -> Self {
        Self {
            name: name.to_string(),
            data: serde_json::to_value(data).unwrap(),
        }
    }

    pub fn parse_data<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_value(self.data.clone())
    }
}

// --- Outbound (snout → overlay) ---

#[derive(Debug, Serialize, Deserialize)]
pub struct InitializePacket {
    #[serde(rename = "AppVersion")]
    pub app_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunVariableLengthRoutinePacket {
    #[serde(rename = "RoutineName")]
    pub routine_name: String,
    #[serde(rename = "Time")]
    pub time: String,
}

impl RunVariableLengthRoutinePacket {
    pub fn new(name: &str, seconds: f64) -> Self {
        let total_secs = seconds as u64;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        let frac = seconds - seconds.floor();
        Self {
            routine_name: name.to_string(),
            time: format!("{hours:02}:{mins:02}:{secs:02}.{:07}", (frac * 10_000_000.0) as u64),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunFixedLengthRoutinePacket {
    #[serde(rename = "RoutineName")]
    pub routine_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainerProgressReportPacket {
    #[serde(rename = "ProgressName")]
    pub progress_name: String,
    #[serde(rename = "CurrentProgress")]
    pub current_progress: i32,
    #[serde(rename = "TargetProgress")]
    pub target_progress: i32,
    #[serde(rename = "Loss")]
    pub loss: f64,
}

// --- Inbound (overlay → snout) ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmdPositionalDataPacket {
    #[serde(rename = "RoutinePitch")]
    pub routine_pitch: f32,
    #[serde(rename = "RoutineYaw")]
    pub routine_yaw: f32,
    #[serde(rename = "RoutineDistance")]
    pub routine_distance: f32,
    #[serde(rename = "RoutineConvergence")]
    pub routine_convergence: f32,
    #[serde(rename = "FovAdjustDistance")]
    pub fov_adjust_distance: f32,
    #[serde(rename = "LeftEyePitch")]
    pub left_eye_pitch: f32,
    #[serde(rename = "LeftEyeYaw")]
    pub left_eye_yaw: f32,
    #[serde(rename = "RightEyePitch")]
    pub right_eye_pitch: f32,
    #[serde(rename = "RightEyeYaw")]
    pub right_eye_yaw: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoutineFinishedPacket {
    #[serde(rename = "RoutineName")]
    pub routine_name: String,
}

#[derive(Debug)]
pub enum OverlayMessage {
    PositionalData(HmdPositionalDataPacket),
    RoutineFinished(String),
    Unknown(String),
}

impl OverlayMessage {
    pub fn from_packet(packet: &Packet) -> Self {
        match packet.name.as_str() {
            "HmdPositionalDataPacket" => {
                match packet.parse_data::<HmdPositionalDataPacket>() {
                    Ok(data) => Self::PositionalData(data),
                    Err(_) => Self::Unknown(packet.name.clone()),
                }
            }
            "RoutineFinishedPacket" => {
                let name = packet
                    .parse_data::<RoutineFinishedPacket>()
                    .map(|p| p.routine_name)
                    .unwrap_or_default();
                Self::RoutineFinished(name)
            }
            _ => Self::Unknown(packet.name.clone()),
        }
    }
}
