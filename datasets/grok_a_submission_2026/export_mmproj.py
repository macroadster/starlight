#!/usr/bin/env python3
"""Export a llama.cpp CLIP mmproj whose one patch reads embedding_head.

Track A (COMPETITION_2026.md) serves ``llama-cli --mmproj --image``. This
file builds that projector for llama.cpp b10150:

* architecture ``clip``, projector type ``mlp``
* the picture is resized the way mtmd resizes a square image (bicubic, 32)
* one patch is a linear map onto the official 64-d ``embedding_head``
* a GELU lift copies those 64 numbers into Qwen2.5-0.5B's 896-d token

The linear map is ridge regression on generated covers. It does not see
alpha, palette indices, EXIF, or JPEG tail bytes, because mtmd feeds RGB
only. It does not contain payload text. ``model/embedding_parity.json``
records cosine against the detector on the public fixtures, which are
held out of the fit.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The script directory is already on sys.path, and the repo root has another
# data_generator.py. Force this folder first so the local generator wins.
for _p in (str(HERE), str(ROOT)):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from data_generator import (  # noqa: E402
    embed_eoi_jpeg,
    embed_exif_jpeg,
    encode_gif,
    encode_png,
    make_cover,
)
from scripts.stego_tool import embed_alpha, embed_lsb, embed_palette  # noqa: E402

IMAGE_SIZE = 32
POOL = 4  # 32 -> 8, so the kernel is piecewise constant
N_VIS = 64
N_HEAD = 4
N_FF = 256
N_LAYER = 2
LLM_DIM = 896
# GELU(x + SHIFT) ~= x + SHIFT for every GELU variant once SHIFT clears the bend.
SHIFT = 32.0
PAD_COLOR = (122, 116, 104)
RIDGE = 10.0


def _clip(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _c_round(v: float) -> int:
    if v >= 0:
        return int(math.floor(v + 0.5))
    return int(math.ceil(v - 0.5))


def resize_bicubic(src: np.ndarray, tw: int, th: int) -> np.ndarray:
    """mtmd ``resize_bicubic`` (tools/mtmd/mtmd-image.cpp, llama.cpp b10150)."""
    ny, nx, _ = src.shape
    dst = np.empty((th, tw, 3), dtype=np.uint8)
    tx = nx / float(tw)
    ty = ny / float(th)
    for i in range(th):
        for j in range(tw):
            x = int(tx * j)
            y = int(ty * i)
            dx = tx * j - x
            dy = ty * i - y
            for k in range(3):
                coeffs = [0.0, 0.0, 0.0, 0.0]

                def px(xx: int, yy: int, ch: int = k) -> float:
                    return float(src[_clip(yy, 0, ny - 1), _clip(xx, 0, nx - 1), ch])

                for jj in range(4):
                    yy = y - 1 + jj
                    d0 = px(x - 1, yy) - px(x, yy)
                    d2 = px(x + 1, yy) - px(x, yy)
                    d3 = px(x + 2, yy) - px(x, yy)
                    a0 = px(x, yy)
                    a1 = (-1.0 / 3.0) * d0 + d2 - (1.0 / 6.0) * d3
                    a2 = 0.5 * d0 + 0.5 * d2
                    a3 = (-1.0 / 6.0) * d0 - 0.5 * d2 + (1.0 / 6.0) * d3
                    coeffs[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx
                d0 = coeffs[0] - coeffs[1]
                d2 = coeffs[2] - coeffs[1]
                d3 = coeffs[3] - coeffs[1]
                a0 = coeffs[1]
                a1 = (-1.0 / 3.0) * d0 + d2 - (1.0 / 6.0) * d3
                a2 = 0.5 * d0 + 0.5 * d2
                a3 = (-1.0 / 6.0) * d0 - 0.5 * d2 + (1.0 / 6.0) * d3
                cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy
                dst[i, j, k] = _c_round(min(max(cc, 0.0), 255.0))
    return dst


def mtmd_square(im: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB float32 in 0..1, matching mtmd's fixed-size path on a square image.

    Mean 0 and std 1 are written into the GGUF, so the tensor is just u8/255.
    Non-square inputs are aspect-fitted and padded with the LLaVA pad color.
    """
    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    ny, nx, _ = rgb.shape
    scale = min(size / float(nx), size / float(ny))
    new_w = min(int(math.ceil(nx * scale)), size)
    new_h = min(int(math.ceil(ny * scale)), size)
    resized = resize_bicubic(rgb, new_w, new_h)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[:] = PAD_COLOR
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    chw = np.transpose(canvas, (2, 0, 1)).astype(np.float64) / 255.0
    return chw


def pool_features(chw: np.ndarray) -> np.ndarray:
    c, h, w = chw.shape
    ph, pw = h // POOL, w // POOL
    blocks = chw.reshape(c, ph, POOL, pw, POOL).mean(axis=(2, 4))
    return blocks.reshape(-1)


def expand_kernel(w_small: np.ndarray) -> np.ndarray:
    """(D, C, 8, 8) average-pool kernel -> (D, C, 32, 32) conv weight."""
    expanded = np.repeat(np.repeat(w_small, POOL, axis=2), POOL, axis=3)
    return expanded / float(POOL * POOL)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_erf(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def project_embedding(emb: np.ndarray) -> np.ndarray:
    """Same affine the GGUF graph applies after the 64-d patch."""
    pre = emb + SHIFT
    # ggml_gelu is the tanh approximation. erf is checked so either backend is close.
    act = gelu_tanh(pre)
    out = np.zeros(LLM_DIM, dtype=np.float64)
    out[:N_VIS] = act - SHIFT
    return out


def embedding_of(model, raw: bytes):
    import torch
    from scripts.starlight_utils import load_unified_input

    pixel, meta, alpha, lsb, palette, palette_lsb, fmt, content = load_unified_input(raw)
    lsb = lsb.permute(2, 0, 1)
    with torch.no_grad():
        out = model(
            pixel.unsqueeze(0),
            meta.unsqueeze(0),
            alpha.unsqueeze(0),
            lsb.unsqueeze(0),
            palette.unsqueeze(0),
            palette_lsb.unsqueeze(0),
            fmt.unsqueeze(0),
            content.unsqueeze(0),
        )
    return out[-1].squeeze(0).detach().cpu().numpy().astype(np.float64)


def load_detector(pth: Path):
    import scanner
    import torch

    model = scanner.BalancedStarlightDetector()
    state = torch.load(pth, map_location="cpu", weights_only=True)
    if "embedding_head.weight" not in state:
        raise SystemExit(f"{pth} has no embedding_head.weight")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _png(im: Image.Image) -> bytes:
    return encode_png(im)


def training_bytes(n_covers: int) -> Iterable[bytes]:
    for i in range(n_covers):
        cover = make_cover(seed=9000 + i, size=256)
        payload = f"track-a-fit-{i}".encode("utf-8")
        yield _png(cover)
        yield _png(embed_alpha(cover, payload))
        yield _png(embed_lsb(cover, payload + b"\x00").convert("RGB"))
        yield encode_gif(embed_palette(cover, payload + b"\x00"))
        yield embed_exif_jpeg(cover, payload)
        yield embed_eoi_jpeg(cover, payload)


def features_of(raw: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(raw))
    return pool_features(mtmd_square(im))


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    n, p = x.shape
    xb = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
    gram = xb.T @ xb
    gram.flat[:: xb.shape[1] + 1] += lam
    gram[-1, -1] -= lam
    coef = np.linalg.solve(gram, xb.T @ y)
    weight = coef[:-1].T  # (D, P)
    bias = coef[-1]
    return weight, bias


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def llm_dim(path: Path) -> int:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    for field in reader.fields.values():
        name = field.name
        if isinstance(name, bytes):
            name = name.decode()
        if name.endswith(".embedding_length"):
            return int(field.contents())
    raise SystemExit(f"{path} has no embedding_length")


def write_gguf(
    dest: Path,
    conv_w: np.ndarray,
    conv_b: np.ndarray,
    llm: int,
) -> None:
    from gguf import GGUFWriter

    if conv_w.shape != (N_VIS, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise SystemExit(f"conv weight {conv_w.shape} is not {(N_VIS, 3, IMAGE_SIZE, IMAGE_SIZE)}")
    if llm != LLM_DIM:
        raise SystemExit(f"language n_embd is {llm}, this projector targets {LLM_DIM}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(dest), "clip")
    writer.add_string("general.name", "starlight-track-a-mmproj")
    writer.add_string(
        "general.description",
        "Starlight Track A projector. One 32x32 patch, fit to embedding_head.",
    )
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_audio_encoder", False)
    writer.add_string("clip.projector_type", "mlp")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.image_size", IMAGE_SIZE)
    writer.add_uint32("clip.vision.patch_size", IMAGE_SIZE)
    writer.add_uint32("clip.vision.embedding_length", N_VIS)
    writer.add_uint32("clip.vision.attention.head_count", N_HEAD)
    writer.add_uint32("clip.vision.feed_forward_length", N_FF)
    writer.add_uint32("clip.vision.block_count", N_LAYER)
    writer.add_uint32("clip.vision.projection_dim", llm)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.0, 0.0, 0.0])
    writer.add_array("clip.vision.image_std", [1.0, 1.0, 1.0])

    def f32(name: str, arr: np.ndarray) -> None:
        writer.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    f32("v.patch_embd.weight", conv_w)
    f32("v.patch_embd.bias", conv_b.astype(np.float32))
    f32("v.position_embd.weight", np.zeros((1, N_VIS), dtype=np.float32))

    # n_layer=2 stores two blocks. b10150 runs the second-to-last (layer 0)
    # and keeps its output. Zero QKV/FFN makes that block a residual identity,
    # so the patch vector reaches the projector unchanged.
    for il in range(N_LAYER):
        f32(f"v.blk.{il}.attn_q.weight", np.zeros((N_VIS, N_VIS), dtype=np.float32))
        f32(f"v.blk.{il}.attn_k.weight", np.zeros((N_VIS, N_VIS), dtype=np.float32))
        f32(f"v.blk.{il}.attn_v.weight", np.zeros((N_VIS, N_VIS), dtype=np.float32))
        f32(f"v.blk.{il}.attn_out.weight", np.zeros((N_VIS, N_VIS), dtype=np.float32))
        f32(f"v.blk.{il}.ffn_up.weight", np.zeros((N_FF, N_VIS), dtype=np.float32))
        f32(f"v.blk.{il}.ffn_down.weight", np.zeros((N_VIS, N_FF), dtype=np.float32))
        f32(f"v.blk.{il}.ln1.weight", np.ones(N_VIS, dtype=np.float32))
        f32(f"v.blk.{il}.ln1.bias", np.zeros(N_VIS, dtype=np.float32))
        f32(f"v.blk.{il}.ln2.weight", np.ones(N_VIS, dtype=np.float32))
        f32(f"v.blk.{il}.ln2.bias", np.zeros(N_VIS, dtype=np.float32))

    f32("mm.0.weight", np.eye(N_VIS, dtype=np.float32))
    f32("mm.0.bias", np.full(N_VIS, SHIFT, dtype=np.float32))
    mm2 = np.zeros((llm, N_VIS), dtype=np.float32)
    mm2[:N_VIS] = np.eye(N_VIS, dtype=np.float32)
    f32("mm.2.weight", mm2)
    mm2_b = np.zeros(llm, dtype=np.float32)
    mm2_b[:N_VIS] = -SHIFT
    f32("mm.2.bias", mm2_b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def collect(model, blobs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i, raw in enumerate(blobs):
        xs.append(features_of(raw))
        ys.append(embedding_of(model, raw))
        if (i + 1) % 12 == 0 or i + 1 == len(blobs):
            print(f"embedded {i + 1}/{len(blobs)}", file=sys.stderr)
    return np.stack(xs), np.stack(ys)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covers", type=int, default=12, help="training covers (x 6 variants)")
    parser.add_argument("--ridge", type=float, default=RIDGE)
    args = parser.parse_args()

    serve = json.loads((HERE / "model" / "serve.json").read_text(encoding="utf-8"))
    llm_rel = serve.get("llm_gguf") or ""
    llm_path = HERE / llm_rel
    if not llm_path.is_file():
        raise SystemExit(f"language GGUF missing at {llm_path}; run fuse_vlm.py first")
    pth = ROOT / serve.get("detector", "models/detector_balanced.pth")
    if not pth.is_file():
        raise SystemExit(f"detector missing at {pth}")
    dim = llm_dim(llm_path)

    print("loading detector", file=sys.stderr)
    model = load_detector(pth)
    blobs = list(training_bytes(args.covers))
    print(f"fitting {len(blobs)} images", file=sys.stderr)
    x, y = collect(model, blobs)
    w_small, bias = fit_ridge(x, y, args.ridge)
    side = IMAGE_SIZE // POOL
    w_small = w_small.reshape(N_VIS, 3, side, side)
    conv_w = expand_kernel(w_small).astype(np.float32)
    conv_b = bias.astype(np.float32)

    train_pred = x @ w_small.reshape(N_VIS, -1).T + bias
    train_cos = [cosine(train_pred[i], y[i]) for i in range(len(y))]

    # The lift should put the 64-d vector in the leading dims of the token.
    probe = y[0]
    lifted = project_embedding(probe)
    lift_err = float(np.max(np.abs(lifted[:N_VIS] - gelu_tanh(probe + SHIFT) + SHIFT)))
    # Compare tanh-GELU reconstruction to the raw embedding. Large SHIFT => ~0.
    recon = gelu_tanh(probe + SHIFT) - SHIFT
    recon_err = float(np.max(np.abs(recon - probe)))
    erf_err = float(np.max(np.abs((gelu_erf(probe + SHIFT) - SHIFT) - probe)))

    held = []
    fixtures = HERE / "eval" / "fixtures"
    if fixtures.is_dir():
        for path in sorted(fixtures.iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
                continue
            raw = path.read_bytes()
            target = embedding_of(model, raw)
            pred = features_of(raw) @ w_small.reshape(N_VIS, -1).T + bias
            held.append(
                {
                    "image": path.name,
                    "cosine": round(cosine(pred, target), 4),
                    "target_l2": round(float(np.linalg.norm(target)), 4),
                    "pred_l2": round(float(np.linalg.norm(pred)), 4),
                }
            )

    report = {
        "track": "A",
        "teacher": "models/detector_balanced.pth",
        "teacher_tensor": "embedding_head",
        "image_size": IMAGE_SIZE,
        "patch_size": IMAGE_SIZE,
        "pool": POOL,
        "ridge_lambda": args.ridge,
        "n_train": int(len(y)),
        "train_cosine_mean": round(float(np.mean(train_cos)), 4),
        "train_cosine_min": round(float(np.min(train_cos)), 4),
        "heldout_cosine_mean": round(float(np.mean([h["cosine"] for h in held])), 4) if held else None,
        "heldout": held,
        "gelu_shift": SHIFT,
        "gelu_tanh_recon_max_abs": recon_err,
        "gelu_erf_recon_max_abs": erf_err,
        "lift_identity_max_abs": lift_err,
        "conv_weight_max_abs": float(np.max(np.abs(conv_w))),
        "note": (
            "Cosine is embedding_head versus the mmproj's linear patch on mtmd RGB. "
            "Clean covers track the teacher when the look is in the pixels. "
            "Alpha mismatches the clean twin because mtmd drops the alpha plane. "
            "EXIF and EOI live outside the pixels. The projector does not store payload text."
        ),
    }
    parity = HERE / "model" / "embedding_parity.json"
    parity.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_train", "train_cosine_mean", "heldout_cosine_mean", "gelu_tanh_recon_max_abs")}, indent=2))

    mmproj_rel = serve.get("mmproj_gguf") or "model/mmproj.gguf"
    dest = HERE / mmproj_rel
    write_gguf(dest, conv_w, conv_b, dim)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    print(f"wrote {parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
