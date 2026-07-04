use crate::calibration::Bounds;
use crate::weights::{Shape, Weights};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EyeShape {
    LeftEyePitch,
    LeftEyeYaw,
    LeftEyeLid,
    LeftEyeWiden,
    LeftEyeBrow,
    LeftEyeSquint,
    RightEyePitch,
    RightEyeYaw,
    RightEyeLid,
    RightEyeWiden,
    RightEyeBrow,
    RightEyeSquint,
}

impl From<EyeShape> for usize {
    fn from(value: EyeShape) -> Self {
        value as usize
    }
}

impl From<usize> for EyeShape {
    fn from(value: usize) -> Self {
        assert!(value < Self::count());

        unsafe { std::mem::transmute(value as u8) }
    }
}

impl Shape for EyeShape {
    fn count() -> usize {
        const {
            assert!(Self::RightEyeSquint as usize + 1 == 12);
        }

        Self::RightEyeSquint as usize + 1
    }
}

impl EyeShape {
    pub const fn count() -> usize {
        const {
            assert!(Self::RightEyeSquint as usize + 1 == 12);
        }

        Self::RightEyeSquint as usize + 1
    }

    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            "rightEyeY" => Some(Self::RightEyePitch),
            "rightEyeX" => Some(Self::RightEyeYaw),
            "rightEyeLid" => Some(Self::RightEyeLid),
            "rightEyeWiden" => Some(Self::RightEyeWiden),
            "rightEyeBrow" => Some(Self::RightEyeBrow),
            "rightEyeSquint" => Some(Self::RightEyeSquint),

            "leftEyeY" => Some(Self::LeftEyePitch),
            "leftEyeX" => Some(Self::LeftEyeYaw),
            "leftEyeLid" => Some(Self::LeftEyeLid),
            "leftEyeWiden" => Some(Self::LeftEyeWiden),
            "leftEyeBrow" => Some(Self::LeftEyeBrow),
            "leftEyeSquint" => Some(Self::LeftEyeSquint),
            _ => None,
        }
    }

    pub(crate) fn to_etvr(self) -> Option<&'static str> {
        match self {
            Self::LeftEyePitch => Some("/avatar/parameters/v2/EyeLeftY"),
            Self::LeftEyeYaw => Some("/avatar/parameters/v2/EyeLeftX"),
            Self::LeftEyeLid => Some("/avatar/parameters/v2/EyeLidLeft"),

            Self::RightEyePitch => Some("/avatar/parameters/v2/EyeRightY"),
            Self::RightEyeYaw => Some("/avatar/parameters/v2/EyeRightX"),
            Self::RightEyeLid => Some("/avatar/parameters/v2/EyeLidRight"),
            _ => None,
        }
    }

    pub(crate) fn to_babble(self) -> &'static str {
        match self {
            Self::LeftEyePitch => "/leftEyeY",
            Self::LeftEyeYaw => "/leftEyeX",
            Self::LeftEyeLid => "/leftEyeLid",
            Self::LeftEyeWiden => "/leftEyeWiden",
            Self::LeftEyeBrow => "/leftEyeBrow",
            Self::LeftEyeSquint => "/leftEyeSquint",

            Self::RightEyePitch => "/rightEyeY",
            Self::RightEyeYaw => "/rightEyeX",
            Self::RightEyeLid => "/rightEyeLid",
            Self::RightEyeWiden => "/rightEyeWiden",
            Self::RightEyeBrow => "/rightEyeBrow",
            Self::RightEyeSquint => "/rightEyeSquint",
        }
    }

    pub(crate) fn to_etvr_value(self, value: f32) -> f32 {
        if self == Self::LeftEyeLid || self == Self::RightEyeLid {
            1. - value
        } else {
            value
        }
    }
}

pub struct EyeCalibrator {
    bounds: Vec<Bounds>,
    weights: Weights<EyeShape>,
    link_eyes: bool,
}

impl EyeCalibrator {
    pub fn new() -> Self {
        let mut bounds = vec![Bounds::new_11(); EyeShape::count()];
        bounds[EyeShape::LeftEyeLid as usize] = Bounds::new_01();
        bounds[EyeShape::RightEyeLid as usize] = Bounds::new_01();

        Self {
            bounds,
            weights: Weights::new(),
            link_eyes: true,
        }
    }

    pub fn link_eyes(&self) -> bool {
        self.link_eyes
    }

    pub fn set_link_eyes(&mut self, link_eyes: bool) {
        self.link_eyes = link_eyes;
    }

    pub fn bounds(&self, shape: EyeShape) -> Bounds {
        self.bounds[shape as usize]
    }

    pub fn set_bounds(&mut self, shape: EyeShape, bounds: Bounds) {
        self.bounds[shape as usize] = bounds;
    }

    pub fn calibrate(&mut self, raw: &Weights<EyeShape>) -> &Weights<EyeShape> {
        self.weights.clear();
        self.remap(raw);
        &self.weights
    }

    fn remap(&mut self, raw: &Weights<EyeShape>) {
        let mul_v = 2.;
        let mul_y = 2.;

        let left_pitch_raw = raw.get(EyeShape::LeftEyePitch).unwrap_or(0.);
        let left_yaw_raw = raw.get(EyeShape::LeftEyeYaw).unwrap_or(0.);
        let left_lid_raw = raw.get(EyeShape::LeftEyeLid).unwrap_or(0.);

        let right_pitch_raw = raw.get(EyeShape::RightEyePitch).unwrap_or(0.);
        let right_yaw_raw = raw.get(EyeShape::RightEyeYaw).unwrap_or(0.);
        let right_lid_raw = raw.get(EyeShape::RightEyeLid).unwrap_or(0.);

        let left_pitch = left_pitch_raw * mul_y - mul_y / 2.;
        let left_yaw = left_yaw_raw * mul_v - mul_v / 2.;
        let left_lid = 1. - left_lid_raw;

        let right_pitch = right_pitch_raw * mul_y - mul_y / 2.;
        let right_yaw = right_yaw_raw * mul_v - mul_v / 2.;
        let right_lid = 1. - right_lid_raw;

        let eye_y = (left_pitch * left_lid + right_pitch * right_lid) / (left_lid + right_lid);

        let mut left_eye_yaw_corrected = right_yaw * (1. - left_lid) + left_yaw * left_lid;
        let mut right_eye_yaw_corrected = left_yaw * (1. - right_lid) + right_yaw * right_lid;

        if self.link_eyes {
            let raw_convergence = (right_eye_yaw_corrected - left_eye_yaw_corrected) / 2.;
            let convergence = raw_convergence.max(0.);

            let average_yaw = (right_eye_yaw_corrected + left_eye_yaw_corrected) / 2.;

            left_eye_yaw_corrected = average_yaw - convergence;
            right_eye_yaw_corrected = average_yaw + convergence;
        }

        self.weights.set(EyeShape::LeftEyeYaw, left_eye_yaw_corrected);
        self.weights.set(EyeShape::LeftEyePitch, eye_y);
        self.weights.set(EyeShape::LeftEyeLid, self.bounds[EyeShape::LeftEyeLid as usize].remap(left_lid));

        self.weights.set(EyeShape::RightEyeYaw, right_eye_yaw_corrected);
        self.weights.set(EyeShape::RightEyePitch, eye_y);
        self.weights.set(EyeShape::RightEyeLid, self.bounds[EyeShape::RightEyeLid as usize].remap(right_lid));

        if let Some(left_eye_widen) = raw.get(EyeShape::LeftEyeWiden) {
            self.weights.set(EyeShape::LeftEyeWiden, left_eye_widen);
        }

        if let Some(right_eye_widen) = raw.get(EyeShape::RightEyeWiden) {
            self.weights.set(EyeShape::RightEyeWiden, right_eye_widen);
        }

        if let Some(left_eye_brow) = raw.get(EyeShape::LeftEyeBrow) {
            self.weights.set(EyeShape::LeftEyeBrow, left_eye_brow);
        }

        if let Some(right_eye_brow) = raw.get(EyeShape::RightEyeBrow) {
            self.weights.set(EyeShape::RightEyeBrow, right_eye_brow);
        }

        if let Some(left_eye_squint) = raw.get(EyeShape::LeftEyeSquint) {
            self.weights.set(EyeShape::LeftEyeSquint, left_eye_squint);
        }

        if let Some(right_eye_squint) = raw.get(EyeShape::RightEyeSquint) {
            self.weights.set(EyeShape::RightEyeSquint, right_eye_squint);
        }
    }
}
