use std::net::{ToSocketAddrs, UdpSocket};

use rosc::{OscMessage, OscPacket, OscType, encoder};
use thiserror::Error;

use crate::calibration::{EyeShape, FaceShape};
use crate::weights::Weights;

pub struct OscTransport {
    socket: UdpSocket,
    destination: std::net::SocketAddr,
}

#[derive(Clone, Debug, Error)]
pub enum TransportError {
    #[error("failed to bind UDP socket")]
    Bind,
    #[error("failed to resolve destination address")]
    Resolve,
}

impl OscTransport {
    pub fn udp(destination: impl ToSocketAddrs) -> Result<Self, TransportError> {
        Ok(Self {
            socket: UdpSocket::bind("0.0.0.0:0").map_err(|_| TransportError::Bind)?,

            destination: destination
                .to_socket_addrs()
                .map_err(|_| TransportError::Resolve)?
                .next()
                .ok_or(TransportError::Resolve)?,
        })
    }

    pub(crate) fn send(&mut self, msg: OscMessage) {
        let msg = OscPacket::Message(msg);

        if let Ok(buf) = encoder::encode(&msg) {
            let _ = self.socket.send_to(&buf, &self.destination);
        }
    }

    // TODO: This should return a TransportError
    pub fn flush(&mut self) -> Result<(), TransportError> {
        // No-op for now
        Ok(())
    }
}

pub struct BabbleEmitter {
    // TODO
}

impl BabbleEmitter {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_face(&mut self, weights: &Weights<FaceShape>, transport: &mut OscTransport) {
        for (shape, value) in weights.iter() {
            let msg = OscMessage {
                addr: shape.to_babble().to_string(),
                args: vec![OscType::Float(value)],
            };

            transport.send(msg);
        }
    }

    pub fn process_eyes(&mut self, weights: &Weights<EyeShape>, transport: &mut OscTransport) {
        for (shape, value) in weights.iter() {
            let msg = OscMessage {
                addr: shape.to_babble().to_string(),
                args: vec![OscType::Float(value)],
            };

            transport.send(msg);
        }
    }
}

pub struct EtvrEmitter {
}

impl EtvrEmitter {
    pub fn new() -> Self {
        Self {}
    }

    pub fn process_eyes(&mut self, weights: &Weights<EyeShape>, transport: &mut OscTransport) {
        for (shape, value) in weights.iter() {
            let value = shape.to_etvr_value(value);
            let Some(addr) = shape.to_etvr() else { continue };

            let msg = OscMessage {
                addr: addr.to_string(),
                args: vec![OscType::Float(value)],
            };

            transport.send(msg);
        }
    }
}

pub struct VrchatEmitter {
    max_yaw_deg: f32,
    max_pitch_deg: f32,
}

impl VrchatEmitter {
    pub fn new(max_yaw_deg: f32, max_pitch_deg: f32) -> Self {
        Self {
            max_yaw_deg,
            max_pitch_deg,
        }
    }

    pub fn process_eyes(&mut self, weights: &Weights<EyeShape>, transport: &mut OscTransport) {
        let left_yaw = weights.get(EyeShape::LeftEyeYaw).unwrap_or(0.).clamp(-1., 1.) * self.max_yaw_deg;
        let right_yaw = weights.get(EyeShape::RightEyeYaw).unwrap_or(0.).clamp(-1., 1.) * self.max_yaw_deg;
        let left_pitch = weights.get(EyeShape::LeftEyePitch).unwrap_or(0.).clamp(-1., 1.) * self.max_pitch_deg;
        let right_pitch = weights.get(EyeShape::RightEyePitch).unwrap_or(0.).clamp(-1., 1.) * self.max_pitch_deg;

        let left_lid = weights.get(EyeShape::LeftEyeLid).unwrap_or(0.);
        let right_lid = weights.get(EyeShape::RightEyeLid).unwrap_or(0.);
        let eyes_closed = 1. - ((left_lid + right_lid) / 2.).clamp(0., 1.);

        let eyes_closed_msg = OscMessage {
            addr: "/tracking/eye/EyesClosedAmount".to_string(),
            args: vec![OscType::Float(eyes_closed)],
        };

        let eye_tracking_msg = OscMessage {
            addr: "/tracking/eye/LeftRightPitchYaw".to_string(),
            args: vec![OscType::Float(-left_pitch),OscType::Float(left_yaw),OscType::Float(-right_pitch),OscType::Float(right_yaw)],
        };

        transport.send(eyes_closed_msg);
        transport.send(eye_tracking_msg);
    }
}
