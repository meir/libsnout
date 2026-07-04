use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};

use super::framer::JsonFramer;
use super::packet::{OverlayMessage, Packet};

const DEFAULT_ADDR: &str = "127.0.0.1:2425";

pub struct OverlayConnection {
    stream: TcpStream,
    framer: JsonFramer,
    read_buf: [u8; 4096],
}

impl OverlayConnection {
    pub fn listen() -> io::Result<Self> {
        Self::listen_on(DEFAULT_ADDR)
    }

    pub fn listen_on(addr: &str) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let (stream, _) = listener.accept()?;
        stream.set_nonblocking(true)?;

        Ok(Self {
            stream,
            framer: JsonFramer::new(),
            read_buf: [0u8; 4096],
        })
    }

    pub fn send(&mut self, packet: &Packet) -> io::Result<()> {
        let json = serde_json::to_string(packet)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.stream.write_all(json.as_bytes())
    }

    pub fn try_recv(&mut self) -> io::Result<Option<OverlayMessage>> {
        match self.stream.read(&mut self.read_buf) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionReset,
                    "overlay disconnected",
                ));
            }
            Ok(n) => {
                let data = std::str::from_utf8(&self.read_buf[..n])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                self.framer.feed(data);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }

        if let Some(json) = self.framer.next() {
            let packet: Packet = serde_json::from_str(&json)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Ok(Some(OverlayMessage::from_packet(&packet)))
        } else {
            Ok(None)
        }
    }
}
