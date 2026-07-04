use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use byteorder::{LittleEndian, WriteBytesExt};

use crate::frame::{FrameMeta, RawFrame};

pub struct CaptureWriter<W: Write> {
    inner: W,
}

impl CaptureWriter<BufWriter<File>> {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

impl<W: Write> CaptureWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner }
    }

    pub fn write_frame(&mut self, frame: &RawFrame) -> io::Result<()> {
        self.write_meta(&frame.meta)?;
        self.inner.write_all(&frame.jpeg_left)?;
        self.inner.write_all(&frame.jpeg_right)?;
        Ok(())
    }

    pub fn write_meta(&mut self, meta: &FrameMeta) -> io::Result<()> {
        self.inner.write_f32::<LittleEndian>(meta.routine_pitch)?;
        self.inner.write_f32::<LittleEndian>(meta.routine_yaw)?;
        self.inner.write_f32::<LittleEndian>(meta.routine_distance)?;
        self.inner
            .write_f32::<LittleEndian>(meta.routine_convergence)?;
        self.inner
            .write_f32::<LittleEndian>(meta.fov_adjust_distance)?;
        self.inner.write_f32::<LittleEndian>(meta.left_eye_pitch)?;
        self.inner.write_f32::<LittleEndian>(meta.left_eye_yaw)?;
        self.inner.write_f32::<LittleEndian>(meta.right_eye_pitch)?;
        self.inner.write_f32::<LittleEndian>(meta.right_eye_yaw)?;
        self.inner
            .write_f32::<LittleEndian>(meta.routine_left_lid)?;
        self.inner
            .write_f32::<LittleEndian>(meta.routine_right_lid)?;
        self.inner
            .write_f32::<LittleEndian>(meta.routine_brow_raise)?;
        self.inner
            .write_f32::<LittleEndian>(meta.routine_brow_angry)?;
        self.inner.write_f32::<LittleEndian>(meta.routine_widen)?;
        self.inner.write_f32::<LittleEndian>(meta.routine_squint)?;
        self.inner.write_f32::<LittleEndian>(meta.routine_dilate)?;

        self.inner.write_i64::<LittleEndian>(meta.timestamp)?;
        self.inner
            .write_i64::<LittleEndian>(meta.video_timestamp_left)?;
        self.inner
            .write_i64::<LittleEndian>(meta.video_timestamp_right)?;

        self.inner
            .write_i32::<LittleEndian>(meta.routine_state.raw())?;
        self.inner.write_i32::<LittleEndian>(meta.jpeg_left_len)?;
        self.inner.write_i32::<LittleEndian>(meta.jpeg_right_len)?;

        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}
