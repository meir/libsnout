#!/usr/bin/env python3
"""Validate a trained dual-eye ONNX against the recorded gaze in a capture .bin.

Builds the deployment input (interleaved L/R temporal stacks) from real frames in
the gaze section and checks whether the predicted gaze tracks the recorded gaze.

Usage:
    python tools/validate_gaze.py --bin user_cal.bin --onnx newtest.onnx
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

FRAME_FMT = "=ffffffffffffffffqqqiii"
FRAME_SIZE = struct.calcsize(FRAME_FMT)  # 100
FIELDS = [
    "routine_pitch",
    "routine_yaw",
    "routine_distance",
    "routine_convergence",
    "fov_adjust_distance",
    "left_eye_pitch",
    "left_eye_yaw",
    "right_eye_pitch",
    "right_eye_yaw",
    "left_lid",
    "right_lid",
    "brow_raise",
    "brow_angry",
    "widen",
    "squint",
    "dilate",
    "timestamp",
    "video_ts_left",
    "video_ts_right",
    "routine_state",
    "jpeg_len_left",
    "jpeg_len_right",
]
FLAG_GOOD_DATA = 1 << 30
FLAG_GAZE_DATA = 1 << 0
GAZE_RANGE_DEG = 45.0
S = 128


def gaze_norm(deg):
    return float(min(1.0, max(0.0, (deg + GAZE_RANGE_DEG) / (2 * GAZE_RANGE_DEG))))


def read_frames(path):
    frames = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(FRAME_SIZE)
            if len(hdr) < FRAME_SIZE:
                break
            d = dict(zip(FIELDS, struct.unpack(FRAME_FMT, hdr)))
            jl, jr = d["jpeg_len_left"], d["jpeg_len_right"]
            if jl <= 0 or jr <= 0 or jl > 10_000_000 or jr > 10_000_000:
                break
            lj, rj = f.read(jl), f.read(jr)
            if len(lj) != jl or len(rj) != jr:
                break
            d["jpeg_left"], d["jpeg_right"] = lj, rj
            frames.append(d)
    return frames


def preprocess(jpeg):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img.shape[:2] != (S, S):
        img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(img).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=300)
    args = ap.parse_args()

    frames = read_frames(args.bin)
    gaze_idx = [
        i
        for i, d in enumerate(frames)
        if (int(d["routine_state"]) & FLAG_GOOD_DATA)
        and (int(d["routine_state"]) & FLAG_GAZE_DATA)
        and i >= 3
    ]
    print(f"{len(frames)} frames, {len(gaze_idx)} usable gaze frames")
    if not gaze_idx:
        raise SystemExit("no gaze frames found")

    pick = gaze_idx[:: max(1, len(gaze_idx) // args.samples)][: args.samples]
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    preds, recs = [], []
    for i in pick:
        # Deployment input: [1, 8, 128, 128], channels interleaved L/R across t0..t3.
        chans = []
        for t in range(4):
            d = frames[i - t]
            chans.append(preprocess(d["jpeg_left"]))  # even channel: left
            chans.append(preprocess(d["jpeg_right"]))  # odd channel: right
        x = np.stack(chans, 0)[None].astype(np.float32)
        out = sess.run(["output"], {"input": x})[0][0]
        preds.append(out)

        d = frames[i]
        recs.append(
            [
                gaze_norm(d["right_eye_pitch"]),
                gaze_norm(d["right_eye_yaw"]),
                gaze_norm(d["left_eye_pitch"]),
                gaze_norm(d["left_eye_yaw"]),
            ]
        )

    preds = np.array(preds)
    recs = np.array(recs)
    np.set_printoptions(precision=4, suppress=True)

    # Predicted gaze channels: 0,1 (right Y,X) and 6,7 (left Y,X).
    pred_gaze = preds[:, [0, 1, 6, 7]]
    print("\npredicted gaze   std :", pred_gaze.std(0), " (near-zero std => collapsed)")
    print("predicted gaze  range:", pred_gaze.max(0) - pred_gaze.min(0))
    print("recorded  gaze   std :", recs.std(0))

    print("\ncorrelation predicted-vs-recorded (per gaze channel):")
    names = ["R.EyeY", "R.EyeX", "L.EyeY", "L.EyeX"]
    for c, name in enumerate(names):
        p, r = pred_gaze[:, c], recs[:, c]
        corr = np.corrcoef(p, r)[0, 1] if p.std() > 1e-6 and r.std() > 1e-6 else 0.0
        mae_deg = float(np.abs(p - r).mean()) * 2 * GAZE_RANGE_DEG
        print(
            f"  {name}: corr={corr:+.3f}  MAE={mae_deg:5.1f}deg  pred[{p.min():.3f},{p.max():.3f}] rec[{r.min():.3f},{r.max():.3f}]"
        )


if __name__ == "__main__":
    main()
