#!/usr/bin/env python3
"""Generate the MergedDualEye ONNX *template* (graph only; weights are placeholders).

The Rust trainer swaps the real trained weights into this template's initializers
at export time, so this script only needs to run once. Module names mirror the
burn module layout (`snout-train`) so the initializer names line up directly.

Plain torch only -- no timm / torchvision.

Usage:
    python tools/export_template.py --out crates/snout-train/assets/merged_template.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNorm(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, (k - 1) // 2, groups=groups, bias=True)

    def forward(self, x):
        return self.conv(x)


class ConvBnAct(ConvNorm):
    def forward(self, x):
        return F.relu(super().forward(x))


class Uib(nn.Module):
    def __init__(self, in_c, out_c, exp, dw_start=None, dw_mid=None, s=1):
        super().__init__()
        start_s, mid_s = (s, 1) if dw_start else (1, s)
        self.dw_start = (
            ConvNorm(in_c, in_c, dw_start, start_s, in_c) if dw_start else None
        )
        self.pw_exp = ConvNorm(in_c, exp, 1)
        self.dw_mid = ConvNorm(exp, exp, dw_mid, mid_s, exp) if dw_mid else None
        self.pw_proj = ConvNorm(exp, out_c, 1)

    def forward(self, x):
        identity = x
        if self.dw_start is not None:
            x = self.dw_start(x)
        x = F.relu(self.pw_exp(x))
        if self.dw_mid is not None:
            x = F.relu(self.dw_mid(x))
        x = self.pw_proj(x)
        if x.shape == identity.shape:
            x = x + identity
        return x


class MobileNetV4(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()
        self.stem = ConvBnAct(in_channels, 32, 3, 2)
        self.blocks = nn.ModuleList(
            [
                ConvBnAct(32, 16, 3, 2),
                ConvBnAct(16, 16, 1, 1),
                ConvBnAct(16, 48, 3, 2),
                ConvBnAct(48, 32, 1, 1),
                Uib(32, 48, 96, dw_start=5, dw_mid=5, s=2),
                Uib(48, 48, 96, dw_mid=3),
                Uib(48, 48, 96, dw_mid=3),
                Uib(48, 48, 96, dw_mid=3),
                Uib(48, 48, 96, dw_mid=3),
                Uib(48, 48, 192, dw_start=3),
                Uib(48, 64, 288, dw_start=3, dw_mid=3, s=2),
                Uib(64, 64, 256, dw_start=5, dw_mid=5),
                Uib(64, 64, 256, dw_mid=5),
                Uib(64, 64, 192, dw_mid=5),
                Uib(64, 64, 256, dw_mid=3),
                Uib(64, 64, 256, dw_mid=3),
                ConvBnAct(64, 480, 1, 1),
            ]
        )
        self.head_conv = ConvBnAct(480, 1280, 1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        x = self.head_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


class MicroChad(nn.Module):
    def __init__(self, out_count=2, in_ch=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 14, 3, 1, 1)
        self.conv2 = nn.Conv2d(14, 21, 3, 1, 1)
        self.conv3 = nn.Conv2d(21, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 47, 3, 1, 1)
        self.conv5 = nn.Conv2d(47, 70, 3, 1, 1)
        self.conv6 = nn.Conv2d(70, 106, 3, 1, 1)
        self.fc_gaze = nn.Linear(106, out_count)

    def forward(self, x):
        for c in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            x = F.max_pool2d(F.relu(c(x)), 2)
        x = F.relu(self.conv6(x))
        x = x.amax(dim=(2, 3))
        return torch.sigmoid(self.fc_gaze(x))


class DualTaskTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaze = MicroChad(2)
        self.expr = MobileNetV4(4, 4)

    def eye(self, x):
        return torch.cat([self.gaze(x), self.expr(x)], dim=1)  # [B, 6]


class MergedDualEye(nn.Module):
    NOFLIP = [1, 3, 5, 7]
    FLIP = [0, 2, 4, 6]

    def __init__(self):
        super().__init__()
        self.tower = DualTaskTower()

    def forward(self, x):
        noflip = x[:, self.NOFLIP, :, :]
        flipped = torch.flip(x[:, self.FLIP, :, :], dims=[-1])
        out_left = self.tower.eye(noflip)
        out_right = self.tower.eye(flipped)
        out_right = torch.cat(
            [out_right[:, 0:1], 1.0 - out_right[:, 1:2], out_right[:, 2:]], dim=1
        )
        return torch.cat([out_right, out_left], dim=1).clamp(0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = MergedDualEye().eval()
    dummy = torch.randn(1, 8, 128, 128)
    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        export_params=True,
        opset_version=18,
        do_constant_folding=False,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Output channel names the inference pipeline maps via EyeShape::from_model_name.
    # Order matches MergedDualEye: [right(6), left(6)], each EyeY/EyeX/EyeLid/EyeWiden/EyeSquint/EyeBrow.
    import json

    import onnx

    sides = ("right", "left")
    per_eye = ["EyeY", "EyeX", "EyeLid", "EyeWiden", "EyeSquint", "EyeBrow"]
    names = [side + e for side in sides for e in per_eye]

    m = onnx.load(str(args.out))
    for key, value in [
        ("blendshape_names", json.dumps(names)),
        ("input_layout", "interleaved_LR_t0..t3_8x128x128_gray01"),
        ("gaze_range_deg", "45.0"),
    ]:
        prop = m.metadata_props.add()
        prop.key, prop.value = key, value

    # Strip the (random) initializer payloads: the Rust trainer overwrites every
    # initializer at export time, so the template only needs names/dims/dtypes.
    # Keeps the shipped asset tiny (~50 KB instead of ~4 MB).
    for init in m.graph.initializer:
        for field in (
            "raw_data",
            "float_data",
            "int32_data",
            "int64_data",
            "double_data",
            "uint64_data",
        ):
            init.ClearField(field)

    onnx.save(m, str(args.out))

    print(
        f"wrote {args.out} (+{len(m.metadata_props)} metadata props, weights stripped)"
    )


if __name__ == "__main__":
    main()
