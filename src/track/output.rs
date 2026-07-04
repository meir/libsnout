use std::net::ToSocketAddrs;

use crate::{
    calibration::{EyeShape, FaceShape},
    config::Config,
    output::{BabbleEmitter, EtvrEmitter, VrchatEmitter, OscTransport, TransportError},
    weights::Weights,
};

pub struct Output {
    pub transport: Option<OscTransport>,
    pub babble: BabbleEmitter,
    pub etvr: EtvrEmitter,
    pub vrchat: VrchatEmitter,
    pub vrchat_transport: Option<OscTransport>,
}

impl Output {
    pub fn new() -> Self {
        Self {
            transport: None,
            babble: BabbleEmitter::new(),
            etvr: EtvrEmitter::new(),
            vrchat: VrchatEmitter::new(30.0, 20.0),
            vrchat_transport: None,
        }
    }

    pub fn with_config(config: &Config) -> Result<Self, TransportError> {
        let mut output = Self::new();

        output.set_destination(&config.output.osc.destination)?;

        if let Some(native) = &config.output.vrchat {
            output.set_native_destination(&native.destination)?;
            output.vrchat = VrchatEmitter::new(native.max_yaw, native.max_pitch);
        }

        Ok(output)
    }

    pub fn set_destination(
        &mut self,
        destination: impl ToSocketAddrs,
    ) -> Result<(), TransportError> {
        self.transport = Some(OscTransport::udp(destination)?);
        Ok(())
    }

    pub fn set_native_destination(
        &mut self,
        destination: impl ToSocketAddrs,
    ) -> Result<(), TransportError> {
        self.vrchat_transport = Some(OscTransport::udp(destination)?);
        Ok(())
    }

    pub fn send_face(&mut self, weights: &Weights<FaceShape>) {
        let Some(transport) = &mut self.transport else {
            return;
        };

        self.babble.process_face(weights, transport);
    }

    pub fn send_eyes(&mut self, weights: &Weights<EyeShape>) {
        let Some(transport) = &mut self.transport else {
            return;
        };

        self.babble.process_eyes(weights, transport);
        self.etvr.process_eyes(weights, transport);

        if let Some(native_transport) = &mut self.vrchat_transport {
            self.vrchat.process_eyes(weights, native_transport);
        }
    }

    pub fn flush(&mut self) -> Result<(), TransportError> {
        if let Some(transport) = &mut self.transport {
            transport.flush()?;
        }

        if let Some(native_transport) = &mut self.vrchat_transport {
            native_transport.flush()?;
        }

        Ok(())
    }
}
