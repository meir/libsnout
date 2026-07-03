use std::io::{self, Write};

use indicatif::MultiProgress;
use tracing_subscriber::fmt::MakeWriter;

/// A tracing writer that keeps log output from clobbering the indicatif
/// status bars.
///
/// Each tracing event arrives as a single newline-terminated `write`, so
/// wrapping it in one `MultiProgress::suspend` cycle mirrors what
/// `indicatif-log-bridge` does per log record: the bars are cleared, the full
/// line is printed, then the bars are redrawn below it. `suspend` also runs
/// even when the draw target is hidden (e.g. output redirected to a file), so
/// logs are never dropped.
#[derive(Clone)]
pub struct StatusLogWriter {
    multi: MultiProgress,
}

impl StatusLogWriter {
    pub fn new(multi: MultiProgress) -> Self {
        Self { multi }
    }
}

impl Write for StatusLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.multi.suspend(|| {
            io::stdout().lock().write_all(buf)?;
            io::Result::Ok(buf.len())
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

impl<'a> MakeWriter<'a> for StatusLogWriter {
    type Writer = StatusLogWriter;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}
