"""Export the speaker encoder to ONNX and prove the browser will agree with the server.

The live streaming and robustness-lab paths run this network in the browser via
onnxruntime-web, because the deployed Space is free-tier CPU and a server round
trip per frame would be unusable. That only works if the exported graph produces
the same embedding as the PyTorch model. If it does not, the browser and the
server disagree about who is speaking and the disagreement is silent.

So the export is not finished until ``--verify`` passes. The mel front-end is
inside the exported graph (see engine/encoder.py), so the browser supplies raw
16 kHz PCM and never has to reimplement feature extraction.

Usage
-----
    python training/export_onnx.py --verify
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from ml_website.engine.encoder import SpeakerEncoder, load_encoder_state
from ml_website.engine.features import SAMPLE_RATE

MODELS_DIR = Path(__file__).resolve().parents[1] / "ml_website" / "models"
DEFAULT_WEB_DIR = Path(__file__).resolve().parents[1] / "frontend" / "public" / "models"

TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "encoder.pt")
    p.add_argument("--out", type=Path, default=MODELS_DIR / "encoder.onnx")
    p.add_argument("--web-dir", type=Path, default=DEFAULT_WEB_DIR, help="Also copy the graph here for Vite to serve.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="Compare ONNX output against PyTorch and fail on drift.")
    p.add_argument("--seconds", type=float, default=3.0, help="Example length used to trace the graph.")
    return p.parse_args()


def load_encoder(checkpoint_path: Path) -> tuple[SpeakerEncoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    model = SpeakerEncoder(
        channels=config.get("channels", 256),
        embedding_dim=config.get("embedding_dim", 192),
    )
    load_encoder_state(model, checkpoint["state_dict"])

    # eval() is load-bearing, not hygiene: it freezes BatchNorm to its running
    # statistics and disables SpecAugment, so the traced graph is deterministic.
    model.eval()
    return model, checkpoint


def verify(model: SpeakerEncoder, onnx_path: Path, seconds: float) -> float:
    """Compare embeddings across several lengths and signal types, return worst drift."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime is required to verify. pip install onnxruntime")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(0)
    cases = {
        "white noise 3s": rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32) * 0.1,
        "white noise 1s": rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1,
        # A different length proves the dynamic axis works, which is what the
        # browser needs when its rolling window is not exactly the traced size.
        "white noise 5.5s": rng.standard_normal(int(SAMPLE_RATE * 5.5)).astype(np.float32) * 0.1,
        "sine 220Hz": np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE).astype(np.float32),
        "near silence": (rng.standard_normal(SAMPLE_RATE * 2) * 1e-4).astype(np.float32),
    }

    worst = 0.0
    print("\nParity check (PyTorch vs ONNX Runtime)")
    for name, audio in cases.items():
        batch = audio[None, :]
        with torch.no_grad():
            expected = model(torch.from_numpy(batch)).numpy()
        actual = session.run(None, {input_name: batch})[0]

        drift = float(np.abs(expected - actual).max())
        cosine = float((expected * actual).sum() / (np.linalg.norm(expected) * np.linalg.norm(actual)))
        worst = max(worst, drift)

        status = "ok" if drift < TOLERANCE else "FAIL"
        print(f"  {name:16s}  max drift {drift:.3e}  cosine {cosine:.8f}  {status}")

    return worst


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Run training/train_encoder.py first.")

    model, checkpoint = load_encoder(args.checkpoint)
    example = torch.randn(1, int(SAMPLE_RATE * args.seconds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(args.out),
        input_names=["waveform"],
        output_names=["embedding"],
        # Both axes are dynamic so the browser can send any window length and
        # the server can batch a whole enrollment in one call.
        dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = args.out.stat().st_size / 1e6
    print(f"Exported {args.out}  ({size_mb:.2f} MB, opset {args.opset})")

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate": SAMPLE_RATE,
        "embedding_dim": model.embedding_dim,
        "input": "raw mono float32 PCM at 16 kHz, any length; mel front-end is inside the graph",
        "output": "L2-normalized embedding; cosine similarity is a plain dot product",
        "cosine_threshold": checkpoint.get("threshold"),
        "heldout_eer": checkpoint.get("eer"),
        "heldout_min_dcf": checkpoint.get("min_dcf"),
        "trained_epoch": checkpoint.get("epoch"),
        "train_speakers": len(checkpoint.get("train_speakers", [])),
        "size_mb": round(size_mb, 2),
    }

    drift = None
    if args.verify:
        drift = verify(model, args.out, args.seconds)
        metadata["max_parity_drift"] = drift
        if drift >= TOLERANCE:
            raise SystemExit(
                f"\nParity check FAILED: worst drift {drift:.3e} exceeds {TOLERANCE:.0e}.\n"
                "The browser would disagree with the server. Not shipping this graph."
            )
        print(f"\nParity ok: worst drift {drift:.3e} < {TOLERANCE:.0e}")

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")

    if args.web_dir:
        args.web_dir.mkdir(parents=True, exist_ok=True)
        (args.web_dir / args.out.name).write_bytes(args.out.read_bytes())
        (args.web_dir / meta_path.name).write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied to {args.web_dir}")


if __name__ == "__main__":
    main()
