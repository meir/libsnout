#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutineState(pub i32);

impl RoutineState {
    pub const FLAG_GOOD_DATA: i32 = 1 << 30;
    pub const FLAG_GAZE_DATA: i32 = 1 << 0;
    pub const FLAG_EXPR_UNLABELED: i32 = 1 << 1;
    pub const FLAG_VERSION_BIT1: i32 = 1 << 20;
    pub const FLAG_IN_MOVEMENT: i32 = 1 << 25;

    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    pub const fn raw(self) -> i32 {
        self.0
    }

    pub const fn is_good_data(self) -> bool {
        self.0 & Self::FLAG_GOOD_DATA != 0
    }

    pub const fn is_gaze_data(self) -> bool {
        self.0 & Self::FLAG_GAZE_DATA != 0
    }

    pub const fn is_expr_unlabeled(self) -> bool {
        self.0 & Self::FLAG_EXPR_UNLABELED != 0
    }
}

impl From<i32> for RoutineState {
    fn from(value: i32) -> Self {
        Self(value)
    }
}

impl From<RoutineState> for i32 {
    fn from(state: RoutineState) -> Self {
        state.0
    }
}
