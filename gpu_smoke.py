#!/usr/bin/env python3
"""Inspect this process's local GPUs and test one small BF16 matrix multiply.

No model is downloaded. No training is performed. Results describe only the
machine running this command, not any other server mentioned in a conversation.
Run with CUDA_VISIBLE_DEVICES=0 or 1 to check each server GPU separately.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical index among visible GPUs")
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        print(json.dumps({"ok": False, "error": "PyTorch is not installed", "python": sys.version}, indent=2))
        return 1

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "num_visible_gpus": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "scope": "Only the local machine executing gpu_smoke.py",
        "devices": [],
    }
    if report["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            with torch.cuda.device(index):
                bf16_kwargs = {}
                if "including_emulation" in inspect.signature(torch.cuda.is_bf16_supported).parameters:
                    bf16_kwargs["including_emulation"] = False
                bf16 = bool(torch.cuda.is_bf16_supported(**bf16_kwargs))
            report["devices"].append({
                "logical_index": index,
                "name": properties.name,
                "total_memory_gib": properties.total_memory / (1024**3),
                "compute_capability": [properties.major, properties.minor],
                "multiprocessor_count": properties.multi_processor_count,
                "native_bf16_supported": bf16 and properties.major >= 8,
            })
    # Print the hardware report before any test allocation or operation.
    print(json.dumps(report, indent=2), flush=True)
    if not report["cuda_available"] or torch.version.cuda is None:
        print(json.dumps({"ok": False, "error": "CUDA-enabled PyTorch and a visible CUDA device are required"}), flush=True)
        return 1
    if not 0 <= args.device < report["num_visible_gpus"]:
        print(json.dumps({"ok": False, "error": "--device is outside the visible GPU range"}), flush=True)
        return 1
    if not report["devices"][args.device]["native_bf16_supported"]:
        print(json.dumps({"ok": False, "error": "Selected GPU does not report native BF16 support"}), flush=True)
        return 1
    try:
        with torch.cuda.device(args.device), torch.no_grad():
            torch.cuda.reset_peak_memory_stats(args.device)
            values = torch.ones((128, 128), device=f"cuda:{args.device}", dtype=torch.bfloat16)
            product = values @ values
            torch.cuda.synchronize(args.device)
            finite = bool(torch.isfinite(product).all().item())
            correct = bool((product == 128).all().item())
            result = {
                "ok": finite and correct,
                "operation": "128x128 BF16 ones matrix @ itself; expected every value = 128",
                "logical_device": args.device,
                "output_dtype": str(product.dtype),
                "finite": finite,
                "expected_result": correct,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(args.device) / (1024**2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(args.device) / (1024**2),
            }
            del product, values
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["ok"] else 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
