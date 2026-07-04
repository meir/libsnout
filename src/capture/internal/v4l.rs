use std::os::fd::RawFd;

use image::GrayImage;
use v4l::video::capture::Parameters;
use v4l::{
    buffer::Type,
    io::traits::CaptureStream,
    prelude::{MmapStream, UserptrStream},
    video::Capture,
};

use crate::capture::internal::Camera;
use crate::capture::sensor::{Gc0308Config, SensorConfig};
use crate::capture::{CameraError, discovery::V4lSource};

#[derive(Copy, Clone, Debug)]
enum PixelFormat {
    Grey,
    Yuyv,
    Uyvy,
    Mjpeg,
}

enum V4lStream {
    Userptr(UserptrStream),
    Mmap(MmapStream<'static>),
}

impl V4lStream {
    fn next(&mut self) -> std::io::Result<(&[u8], &v4l::buffer::Metadata)> {
        match self {
            V4lStream::Userptr(s) => s.next(),
            V4lStream::Mmap(s) => s.next(),
        }
    }
}

struct PendingSensor {
    config: SensorConfig,
    attempts: u32,
    warmup: u32,
}

pub struct V4lCamera {
    device: v4l::Device,
    stream: V4lStream,
    pixel_format: PixelFormat,
    index: u8,
    pub width: usize,
    pub height: usize,

    /// Sensor config waiting to be applied once the stream is live.
    pending_sensor: Option<PendingSensor>,
}

impl Camera for V4lCamera {
    fn read_frame(&mut self) -> Result<GrayImage, CameraError> {
        let mut destination = GrayImage::new(self.width as _, self.height as _);
        self.read_frame(&mut destination)?;
        Ok(destination)
    }

    fn set_sensor_config(&mut self, config: SensorConfig) {
        // Gate on the hardware once, up front, so read_frame stays a no-op on
        // cameras this config doesn't apply to.
        let compatible = match &config {
            SensorConfig::Gc0308(_) => gc0308::is_compatible_target(self.index),
        };

        if compatible {
            self.pending_sensor = Some(PendingSensor {
                config,
                attempts: 5,
                warmup: 5,
            });
        } else {
            tracing::debug!(index = self.index, "sensor config ignored: not a compatible target");
        }
    }
}

impl V4lCamera {
    pub fn open(source: V4lSource) -> Result<Self, CameraError> {
        tracing::debug!(
            index = source.index,
            width = source.format.width,
            height = source.format.height,
            fps = source.format.fps,
            fourcc = %String::from_utf8_lossy(&source.fourcc),
            "Opening V4L2 device"
        );

        let device = v4l::Device::new(source.index as usize)?;

        let format = v4l::Format::new(
            source.format.width,
            source.format.height,
            v4l::FourCC::new(&source.fourcc),
        );
        let format = device.set_format(&format)?;

        tracing::debug!(
            width = format.width,
            height = format.height,
            "Format negotiated"
        );

        let params = Parameters::with_fps(source.format.fps);
        device.set_params(&params)?;

        let pixel_format = match &source.fourcc {
            b"GREY" => PixelFormat::Grey,
            b"YUYV" => PixelFormat::Yuyv,
            b"UYVY" => PixelFormat::Uyvy,
            b"MJPG" => PixelFormat::Mjpeg,
            _ => {
                return Err(CameraError::InvalidFormat(format!(
                    "Unknown pixel format: {:?}",
                    &source.fourcc
                )));
            }
        };

        let width = format.width as usize;
        let height = format.height as usize;

        let stream = match UserptrStream::new(&device, Type::VideoCapture) {
            Ok(s) => {
                tracing::debug!("Using userptr streaming mode");
                V4lStream::Userptr(s)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Userptr streaming not supported, falling back to mmap");
                let s = MmapStream::new(&device, Type::VideoCapture)?;
                tracing::debug!("Using mmap streaming mode");
                V4lStream::Mmap(s)
            }
        };

        Ok(Self {
            device,
            stream,
            pixel_format,
            index: source.index,
            width,
            height,
            pending_sensor: None,
        })
    }

    pub fn read_frame(&mut self, destination: &mut GrayImage) -> Result<(), CameraError> {
        let (buf, _meta) = self.stream.next().map_err(|e| {
            tracing::error!(
                error = %e,
                pixel_format = ?self.pixel_format,
                width = self.width,
                height = self.height,
                "Failed to read frame from V4L2 stream"
            );
            e
        })?;
        match self.pixel_format {
            PixelFormat::Grey => destination.copy_from_slice(buf),
            PixelFormat::Yuyv => {
                // extract Y channel: every other byte
                for (dst, &y) in destination.iter_mut().zip(buf.iter().step_by(2)) {
                    *dst = y;
                }
            }
            PixelFormat::Uyvy => {
                for (dst, &y) in destination.iter_mut().zip(buf[1..].iter().step_by(2)) {
                    *dst = y;
                }
            }
            PixelFormat::Mjpeg => {
                let img = image::load_from_memory(&buf[..])
                    .map_err(|e| CameraError::InvalidFrame(e.to_string()))?
                    .into_luma8();
                destination.copy_from_slice(img.as_raw());
            }
        }

        self.flush_sensor_config();

        Ok(())
    }

    /// Applies a pending sensor config once the stream has warmed up, retrying
    /// on failure (the sensor NAKs I2C until it is powered). A single blocking
    /// I2C burst on the frame it succeeds, then nothing.
    fn flush_sensor_config(&mut self) {
        let Some(pending_sensor) = self.pending_sensor.as_mut() else { return; };

        if pending_sensor.warmup > 0 {
            pending_sensor.warmup -= 1;
            return;
        }

        let fd = self.device.handle().fd();
        let result = match &pending_sensor.config {
            SensorConfig::Gc0308(cfg) => write_gc0308(fd, cfg),
        };

        match result {
            Ok(()) => {
                tracing::info!(
                    config = ?pending_sensor.config,
                    "applied sensor config"
                );
            },
            // The USB id only identifies the Sonix bridge, so a matching board
            // may carry a different sensor. A chip-id mismatch is definitive:
            // don't retry.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(error = %e, "sensor did not identify as expected, ignoring config");
            }
            Err(e) => {
                pending_sensor.attempts -= 1;

                if pending_sensor.attempts > 0 {
                    tracing::debug!(
                        error = %e,
                        remaining = pending_sensor.attempts,
                        "sensor config not ready, retrying",
                    );
                    return;
                } else {
                    tracing::warn!(error = %e, "giving up applying sensor config");
                }
            }
        };

        self.pending_sensor = None;
    }
}

/// Applies a GC0308 config: disables the on-sensor auto-exposure loop (which
/// also writes the sensor defaults) and then overrides with any explicit
/// exposure/gain. Uses the [`gc0308`] crate as-is.
fn write_gc0308(fd: RawFd, cfg: &Gc0308Config) -> std::io::Result<()> {
    gc0308::disable_aec(fd)?;

    gc0308::set_exposure(fd, cfg.exposure)?;
    gc0308::set_gain(fd, cfg.gain)?;

    Ok(())
}
