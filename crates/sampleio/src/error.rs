use std::io;

#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid JPEG length: left={left}, right={right}")]
    InvalidJpegLength { left: i32, right: i32 },

    #[error("JPEG length exceeds 10 MiB guard: left={left}, right={right}")]
    JpegTooLarge { left: i32, right: i32 },
}
