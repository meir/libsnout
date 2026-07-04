//! Direct GC0308 sensor control over the Sonix/Microdia SN9C29x UVC bridge.
//!
//! These cheap GC0308 boards (e.g. `0c45:6366`, "USB 2.0 Camera") expose *no*
//! working V4L2 exposure/gain controls. The sensor's internal AEC owns those
//! registers and silently overwrites anything set through UVC. The only way to
//! get deterministic exposure is to talk to the sensor's I2C registers directly
//! through the bridge's vendor extension unit (XU, GUID `28f03370-...`, unit 3).
//!
//! Protocol ported from Kurokesu's `C1_SONIX_Test_AP` (`sonix_xu_ctrls.c`):
//! ASIC register R/W via XU selector `0x01`, and the I2C master driven through
//! ASIC registers `0x10d0..0x10d9`.
//!
//! The sensor only ACKs I2C while the stream is active (it is power-gated
//! otherwise), so [`apply`] must be called *after* `STREAMON`.

use std::borrow::Cow;
use std::os::raw::c_int;
use std::time::Duration;

// UVC query codes (linux/uvcvideo.h)
const UVC_SET_CUR: u8 = 0x01;
const UVC_GET_CUR: u8 = 0x81;

const XU_UNIT: u8 = 3;
const XU_ASIC_RW: u8 = 0x01;

const GC0308_SLAVE: u8 = 0x21;
const GC0308_CHIP_ID_REG: u8 = 0x00;
const GC0308_CHIP_ID: u8 = 0x9b;
const GC0308_AEC_REG: u8 = 0xd2; // bit7 = AEC enable
const GC0308_EXP_HI: u8 = 0x03; // [3:0] = exposure[11:8]
const GC0308_EXP_LO: u8 = 0x04; // exposure[7:0]
const GC0308_GAIN: u8 = 0x50; // global gain

const I2C_WRITE: u8 = 0;
const I2C_READ: u8 = 1;

/// Default fixed exposure applied when the auto-exposure loop is disabled.
pub const DEFAULT_EXPOSURE: u16 = 384;
/// Default fixed global gain applied when the auto-exposure loop is disabled.
pub const DEFAULT_GAIN: u8 = 0x10;

#[derive(PartialEq, Eq, Clone, Debug)]
struct UsbId {
    pid: Cow<'static, str>,
    vid: Cow<'static, str>,
}

impl UsbId {
    /// The one USB device this control applies to (Sonix bridge + GC0308). Hardcoded
    /// because the protocol and register map are specific to this exact board.
    const GC0308: Self = Self {
        vid: Cow::Borrowed("0c45"),
        pid: Cow::Borrowed("6366"),
    };
}

/// Reads the USB `vid:pid` (lowercase hex) for a V4L2 device index from sysfs,
/// e.g. `"0c45:6366"`. Returns `None` for non-USB devices or on any read error.
fn usb_id(video_index: u8) -> Option<UsbId> {
    let base = format!("/sys/class/video4linux/video{video_index}/device/..");
    let vid = std::fs::read_to_string(format!("{base}/idVendor")).ok()?;
    let pid = std::fs::read_to_string(format!("{base}/idProduct")).ok()?;

    Some(UsbId {
        vid: Cow::Owned(vid.trim().to_lowercase()),
        pid: Cow::Owned(pid.trim().to_lowercase()),
    })
}

/// Whether the device at `video_index` is the supported GC0308/Sonix board.
/// This hardware gate runs before any I2C traffic, so the sensor control is a
/// silent no-op on every other camera.
pub fn is_compatible_target(video_index: u8) -> bool {
    usb_id(video_index) == Some(UsbId::GC0308)
}

#[repr(C)]
struct UvcXuControlQuery {
    unit: u8,
    selector: u8,
    query: u8,
    size: u16,
    data: *mut u8,
}

const _: () = assert!(std::mem::size_of::<UvcXuControlQuery>() == 16);

// _IOWR('u', 0x21, struct uvc_xu_control_query) for the 16-byte struct.
const UVCIOC_CTRL_QUERY: u64 = (3 << 30) | (16 << 16) | ((b'u' as u64) << 8) | 0x21;

fn xu_query(fd: c_int, selector: u8, query: u8, data: &mut [u8]) -> std::io::Result<()> {
    let mut q = UvcXuControlQuery {
        unit: XU_UNIT,
        selector,
        query,
        size: data.len() as u16,
        data: data.as_mut_ptr(),
    };
    let ret = unsafe {
        libc::ioctl(
            fd,
            UVCIOC_CTRL_QUERY as _,
            &mut q as *mut UvcXuControlQuery as *mut libc::c_void,
        )
    };
    if ret < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn asic_write(fd: c_int, addr: u16, val: u8) -> std::io::Result<()> {
    let mut d = [(addr & 0xff) as u8, (addr >> 8) as u8, val, 0x00];
    xu_query(fd, XU_ASIC_RW, UVC_SET_CUR, &mut d)
}

fn asic_read(fd: c_int, addr: u16) -> std::io::Result<u8> {
    // dummy write latches the address, then GET_CUR returns the value in byte[2]
    let mut d = [(addr & 0xff) as u8, (addr >> 8) as u8, 0x00, 0xFF];
    xu_query(fd, XU_ASIC_RW, UVC_SET_CUR, &mut d)?;
    let mut out = [0u8; 4];
    xu_query(fd, XU_ASIC_RW, UVC_GET_CUR, &mut out)?;
    Ok(out[2])
}

/// Drives the bridge's I2C master (registers `0x10d0..0x10d9`).
fn i2c_cmd(fd: c_int, cmd: u8, slave: u8, data: &[u8; 5], len: u8) -> std::io::Result<[u8; 5]> {
    asic_write(fd, 0x10d9, 1)?;
    asic_write(fd, 0x10d8, 1)?;
    asic_write(fd, 0x10d0, 0x80 | ((len & 0x7) << 4) | (cmd << 1))?;
    asic_write(fd, 0x10d1, slave)?;
    for (i, &b) in data.iter().enumerate() {
        asic_write(fd, 0x10d2 + i as u16, b)?;
    }
    asic_write(fd, 0x10d7, 0x10)?; // trigger
    std::thread::sleep(Duration::from_millis(5));

    let mut status = 0u8;
    for _ in 0..30 {
        status = asic_read(fd, 0x10d0)?;
        if status & 0x4 != 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if status & 0x4 == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "GC0308 I2C transaction did not complete",
        ));
    }
    if status & 0x8 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "GC0308 I2C transaction NAK",
        ));
    }

    let mut out = [0u8; 5];
    if cmd == I2C_READ {
        for (i, b) in out.iter_mut().enumerate() {
            *b = asic_read(fd, 0x10d2 + i as u16)?;
        }
    }
    Ok(out)
}

fn i2c_read_reg(fd: c_int, reg: u8) -> std::io::Result<u8> {
    i2c_cmd(fd, I2C_WRITE, GC0308_SLAVE, &[reg, 0, 0, 0, 0], 1)?; // send register address
    let out = i2c_cmd(fd, I2C_READ, GC0308_SLAVE, &[0; 5], 1)?; // read 1 byte
    Ok(out[4]) // 1-byte read is right-aligned in the 5-byte field
}

fn i2c_write_reg(fd: c_int, reg: u8, val: u8) -> std::io::Result<()> {
    i2c_cmd(fd, I2C_WRITE, GC0308_SLAVE, &[reg, val, 0, 0, 0], 2)?;
    Ok(())
}

/// Applies the fixed configuration to the GC0308: disables the internal AEC
/// (required for manual exposure) and writes the fixed exposure + gain. Gated on
/// the USB id and chip ID, so it is a safe no-op on any other camera. Must be
/// called while streaming.
///
/// Make sure to call [`is_target`] first to check if the target camera is a GC0308.
pub fn disable_aec(fd: c_int) -> std::io::Result<()> {
    let id = i2c_read_reg(fd, GC0308_CHIP_ID_REG)?;

    if id != GC0308_CHIP_ID {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("sensor is not a GC0308 (chip id 0x{id:02x})"),
        ));
    }

    let aec = i2c_read_reg(fd, GC0308_AEC_REG)?;
    i2c_write_reg(fd, GC0308_AEC_REG, aec & 0x7f)?; // clear bit7 (AEC enable)

    set_exposure(fd, DEFAULT_EXPOSURE)?;
    set_gain(fd, DEFAULT_GAIN)?;

    Ok(())
}

/// Sets the fixed exposure live (e.g. from a GUI). No-op on unsupported cameras.
/// Only takes effect once AEC is disabled (which [`apply`] does on open).
pub fn set_exposure(fd: c_int, exposure: u16) -> std::io::Result<()> {
    let exposure = exposure.min(0x0fff); // 12-bit

    i2c_write_reg(fd, GC0308_EXP_HI, ((exposure >> 8) & 0x0f) as u8)?;
    i2c_write_reg(fd, GC0308_EXP_LO, (exposure & 0xff) as u8)
}

/// Sets the fixed global gain live (e.g. from a GUI). No-op on unsupported cameras.
pub fn set_gain(fd: c_int, gain: u8) -> std::io::Result<()> {
    i2c_write_reg(fd, GC0308_GAIN, gain)
}
