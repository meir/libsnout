//! Building temporal-stack sample indices and partitioning them into training pools.

use crate::data::capture::Frame;
use crate::data::label::Expr;
use crate::spec::TEMPORAL_DEPTH;

const CLOSED_LID_THRESHOLD: f32 = 0.5;
const ACTIVE_EXPR_THRESHOLD: f32 = 0.5;

/// A temporal stack of frame indices into the flat frame pool (newest first).
pub type Sample = [usize; TEMPORAL_DEPTH];

/// Builds a temporal stack for the frame at `idx`.
/// Frames are interleaved [left, right, left, right, ...], so same-eye predecessors
/// are 2 apart.
fn stack(idx: usize) -> Option<Sample> {
    if idx < 6 {
        return None;
    }
    Some([idx, idx - 2, idx - 4, idx - 6])
}

fn is_active(expr: &Expr) -> bool {
    expr.lid > ACTIVE_EXPR_THRESHOLD
        || expr.widen > ACTIVE_EXPR_THRESHOLD
        || expr.squint > ACTIVE_EXPR_THRESHOLD
        || expr.brow > ACTIVE_EXPR_THRESHOLD
}

/// Collects a temporal stack for every frame whose `(index, frame)` satisfies `keep`.
fn collect(frames: &[Frame], keep: impl Fn(usize, &Frame) -> bool) -> Vec<Sample> {
    frames
        .iter()
        .enumerate()
        .filter(|&(i, frame)| keep(i, frame))
        .filter_map(|(i, _)| stack(i))
        .collect()
}

/// Neutral gaze: gaze-valid, expression-valid, no active expression.
/// These are the plain "look at the dot" frames.
pub fn neutral_gaze(frames: &[Frame]) -> Vec<Sample> {
    collect(frames, |_, f| {
        f.gaze.is_some() && f.expr.is_some_and(|e| !is_active(&e))
    })
}

/// Expression-during-gaze: gaze-valid AND (active expression OR free expression).
/// Covers widen/squint/brow-with-reticle and the free expression pass.
pub fn expr_gaze(frames: &[Frame]) -> Vec<Sample> {
    collect(frames, |_, f| {
        f.gaze.is_some() && (f.expr.is_none() || f.expr.is_some_and(|e| is_active(&e)))
    })
}

/// Closed-eye frames: not gaze-valid, expression-valid, lid above threshold.
/// Used for gaze regularization (the batcher provides a center-gaze target).
pub fn closed(frames: &[Frame]) -> Vec<Sample> {
    collect(frames, |_, f| {
        f.gaze.is_none() && f.expr.is_some_and(|e| e.lid > CLOSED_LID_THRESHOLD)
    })
}

/// Free expression frames (expr is None). Used for anti-jitter training.
pub fn free(frames: &[Frame]) -> Vec<Sample> {
    collect(frames, |_, f| f.expr.is_none())
}

/// Paired-eye samples filtered by `keep(left_expr, right_expr)`: returns `[left, right]`
/// for each pair where both eyes have valid expressions and the predicate holds.
fn paired_with(frames: &[Frame], keep: impl Fn(&Expr, &Expr) -> bool) -> Vec<[Sample; 2]> {
    let mut out = Vec::new();

    for p in 0..(frames.len() / 2) {
        let (Some(left), Some(right)) = (frames[p * 2].expr, frames[p * 2 + 1].expr) else {
            continue;
        };
        if !keep(&left, &right) {
            continue;
        }
        let (Some(ls), Some(rs)) = (stack(p * 2), stack(p * 2 + 1)) else {
            continue;
        };
        out.push([ls, rs]);
    }

    out
}

/// Paired-eye samples with an active expression in at least one eye (the pairs the
/// Python paired `active_boost` oversamples).
pub fn paired_active(frames: &[Frame]) -> Vec<[Sample; 2]> {
    paired_with(frames, |l, r| is_active(l) || is_active(r))
}

/// Paired-eye samples with neither eye active (the neutral L/R pairs).
pub fn paired_neutral(frames: &[Frame]) -> Vec<[Sample; 2]> {
    paired_with(frames, |l, r| !is_active(l) && !is_active(r))
}

/// Pairs consecutive same-eye stacks for anti-jitter / Mean-Teacher training.
/// Returns `[current, previous]` for each valid pair.
pub fn temporal(frames: &[Frame]) -> Vec<[Sample; 2]> {
    let mut out = Vec::new();
    let mut prev_left: Option<Sample> = None;
    let mut prev_right: Option<Sample> = None;

    for p in 0..(frames.len() / 2) {
        if frames[p * 2].expr.is_some() {
            continue;
        }

        let left_stack = stack(p * 2);
        let right_stack = stack(p * 2 + 1);

        if let (Some(cur), Some(prv)) = (left_stack, prev_left) {
            out.push([cur, prv]);
        }
        if let (Some(cur), Some(prv)) = (right_stack, prev_right) {
            out.push([cur, prv]);
        }

        prev_left = left_stack.or(prev_left);
        prev_right = right_stack.or(prev_right);
    }

    out
}
