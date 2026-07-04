#!/usr/bin/env python3
"""Convert the Babble PyTorch checkpoints to burn-compatible safetensors.

Remaps the timm `mobilenetv4_conv_small` (expression net) and MicroChad (gaze net)
key names onto the burn module layout used by `mobilenetv4-burn` and
`snout-train::eye_net`, then writes `.safetensors`.

Requires: torch, safetensors, numpy (see tools/requirements.txt).

Usage:
    python tools/convert_checkpoints.py \
        --expr model_best.pt --gaze gaze_model_best.pt --out-dir crates/snout-train/assets
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

# timm BatchNorm param name -> burn BatchNorm param name
BN_SUFFIX = {
    "weight": "gamma",
    "bias": "beta",
    "running_mean": "running_mean",
    "running_var": "running_var",
}


def pick(ckpt, keys):
    if isinstance(ckpt, dict):
        for k in keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def remap_expr(sd):
    """timm mobilenetv4_conv_small -> mobilenetv4-burn `MobileNetV4`."""
    # Group keys per (stage, block) to detect the block variant + assign a flat index.
    blocks = {}
    for key in sd:
        if key.startswith("blocks."):
            _, s, b, *_ = key.split(".")
            blocks.setdefault((int(s), int(b)), set()).add(key)

    def variant(keys):
        uib_parts = ("dw_start", "pw_exp", "pw_proj", "dw_mid")
        return "Uib" if any(p in k for k in keys for p in uib_parts) else "ConvBnAct"

    order = sorted(blocks)
    flat = {sb: i for i, sb in enumerate(order)}
    variants = {sb: variant(blocks[sb]) for sb in order}

    out = {}
    for key, tensor in sd.items():
        if key.endswith("num_batches_tracked"):
            continue
        nk = None
        if key.startswith("conv_stem."):
            nk = "stem.conv." + key[len("conv_stem.") :]
        elif key.startswith("bn1."):
            nk = "stem.bn." + BN_SUFFIX[key[len("bn1.") :]]
        elif key.startswith("conv_head."):
            nk = "head_conv.conv." + key[len("conv_head.") :]
        elif key.startswith("norm_head."):
            nk = "head_conv.bn." + BN_SUFFIX[key[len("norm_head.") :]]
        elif key.startswith("classifier."):
            nk = key
            if key == "classifier.weight":
                tensor = tensor.t()  # PyTorch Linear [out, in] -> burn [in, out]
        elif key.startswith("blocks."):
            parts = key.split(".")
            sb = (int(parts[1]), int(parts[2]))
            idx, var = flat[sb], variants[sb]
            rest = parts[3:]
            if var == "ConvBnAct":
                if rest[0] == "conv":
                    nk = f"blocks.{idx}.ConvBnAct.conv." + ".".join(rest[1:])
                elif rest[0] == "bn1":
                    nk = f"blocks.{idx}.ConvBnAct.bn." + BN_SUFFIX[rest[1]]
            else:
                part = rest[0]  # dw_start / pw_exp / dw_mid / pw_proj
                if rest[1] == "conv":
                    nk = f"blocks.{idx}.Uib.{part}.conv." + ".".join(rest[2:])
                elif rest[1] == "bn":
                    nk = f"blocks.{idx}.Uib.{part}.bn." + BN_SUFFIX[rest[2]]
        if nk is None:
            raise SystemExit(f"unmapped expr key: {key}")
        out[nk] = tensor.contiguous()
    return out


def remap_gaze(sd):
    """MicroChad keys (conv1..conv6, fc_gaze) already match EyeNet field names."""
    out = {}
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            continue
        if k == "fc_gaze.weight":
            v = v.t()  # PyTorch Linear [out, in] -> burn [in, out]
        out[k] = v.contiguous()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expr", type=Path, required=True)
    ap.add_argument("--gaze", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expr_ckpt = torch.load(args.expr, map_location="cpu", weights_only=True)
    expr = remap_expr(pick(expr_ckpt, ("teacher", "student")))
    save_file(expr, str(args.out_dir / "expr_pretrain.safetensors"))
    print(f"expr: {len(expr)} tensors -> {args.out_dir / 'expr_pretrain.safetensors'}")

    gaze_ckpt = torch.load(args.gaze, map_location="cpu", weights_only=True)
    gaze = remap_gaze(pick(gaze_ckpt, ("gaze", "student")))
    save_file(gaze, str(args.out_dir / "gaze_pretrain.safetensors"))
    print(f"gaze: {len(gaze)} tensors -> {args.out_dir / 'gaze_pretrain.safetensors'}")


if __name__ == "__main__":
    main()
