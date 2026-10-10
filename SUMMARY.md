# YOLO26 Vulkan Inference Engine — Summary

A from-scratch Vulkan compute inference engine for **YOLO26** (n / s / m) written in C + GLSL,
built on the proven Pumice LLM engine core (verbatim Vulkan buffer/device/descriptor/pipeline/
session/dispatch layer, `operation` dispatch API, fp16 block-16 weight layout, RX 580 wave64
16×16/256-thread GEMM tiling). FP16 weights with FP32 accumulation; fixed 640×640 input.

The engine reads every tensor dimension at runtime from the exported graph, so the **same binary**
runs n, s and m without recompilation.

## Performance (RX 580 8 GB, 640×640, batch 1)

Measured with `bin/test.exe <model> --shader bin/shader --bench 200` (200 warm frames; wall-clock
per frame including input upload, op recording, queue submit and fence wait):

| Model | Classes | Ops | Weights | GPU time | **Per-frame** | **FPS** |
|---|---|---|---|---|---|---|
| **YOLO26n** | 80 | 274 | 5.2 MB | 10.4 ms | **12.4 ms** | **80.6** |
| **YOLO26s** | 4  | 302 | 19.2 MB | 29.8 ms | **31.4 ms** | **31.9** |
| **YOLO26m** | 80 | 550 | 41.1 MB | 83.5 ms | **87.4 ms** | **11.4** |

End-to-end through the Python `ctypes` wrapper (includes letterbox + CHW conversion + readback +
CPU decode), 50 frames:

| Model | Per-frame | FPS |
|---|---|---|
| n | 23.6 ms | 42.5 |
| s | 42.1 ms | 23.8 |
| m | 135.7 ms | 7.4 |

GPU time breakdown (`--timing`, op-time only, sum of per-op timestamps):

| Kernel | n | s | m |
|---|---|---|---|
| CONV-GEMM-Q16 (3×3 im2col GEMM) | 4.4 ms | 12.9 ms | 43.1 ms |
| Im2col | 2.7 ms | 5.2 ms | 15.3 ms |
| CONV1x1-Q16 (1×1 direct GEMM) | 2.4 ms | 7.8 ms | 20.8 ms |
| DWConv / MaxPool / Upsample | 0.3 ms | 0.8 ms | 1.0 ms |
| Attention (FF32 GEMM + Softmax + Transpose) | 0.3 ms | 1.1 ms | 0.5 ms |
| Elementwise (SiLU-add, concat copies, head decode) | 0.4 ms | 1.9 ms | 2.8 ms |

## Correctness

Validated against ONNX Runtime on the same random input, comparing the full pre-decode prediction
tensor `[1, 8, 8400]` (4 box channels + classes) at FP16 tolerance:

| Model | box max abs err (px) | class max abs err |
|---|---|---|
| n | 0.37 | 0.016 |
| s | 0.38 | 0.007 |
| m | 0.81 | 0.004 |

The pipeline is also checked layer-by-layer against ONNX Runtime intermediate activations (stem
SiLU, C3k2 concat, SPPF, C2PSA attention output, head box/class logits) — all within FP16 rounding.

## Architecture

### Export pipeline (`tools/export/`, uv venv, one-time build dependency)
1. `export_yolo26.py`-style export: `.pt` → ONNX (`imgsz=640, nms=False, dynamic=False, opset=14,
   simplify=True`). Ultralytics emits the end-to-end one-to-one head; the one-to-many branch is
   dead-code-eliminated by the exporter.
2. `parse_onnx.py`: walks the frozen ONNX graph, fuses `Conv → Sigmoid → Mul` into a SiLU epilogue,
   lowers each node to explicit engine ops, and emits `graph.json` + `weights.bin`.
   - Conv 3×3 → `Im2col` + `CONV-GEMM-Q16`; Conv 1×1 → `CONV1x1-Q16` (reads NCHW directly as the
     transposed A-operand); grouped Conv → `DWConv`.
   - `Split`/`Reshape`/`Slice` are free metadata (buffer views + offsets); `Concat` is materialized
     by channel-block or per-channel copies (handles both channel-axis and anchor-axis concat).
   - C2PSA attention is lowered to per-head fp32 matmuls (`CONV-GEMM-FF32`), softmax and 2-D
     transposes — the qkv tensor is read in place via head/part offsets.
   - Head box decode (`Sub`/`Add`/`Mul` by baked anchor-grid constants) + class `Sigmoid` run on GPU;
     the engine's single readback tensor is the `[1, C+4, 8400]` pre-top-k prediction.
   - `weights.bin`: fp16 block-16 (K-major) GEMM weights, raw fp16 depthwise weights, fp32 biases and
     head constants. K padded to `ceil16`, N padded to `ceil32`.
3. `dump_reference.py` / `validate.py`: ONNX Runtime reference extraction + comparison.
4. `inspect_onnx.py`: graph/node/attribute dump used while reverse-engineering the head.

### Engine core (`src/`, `include/`)
- `buffer/device/descriptor/pipeline/session/fence/dispatch/json.c` — verbatim from Pumice.
- `graph.c` — parses `graph.json`, allocates one VRAM buffer per activation and per weight, builds
  `operation[]` (shader, bindings, push constants, dispatch), uploads the input, runs `execute()`.
  The output tensor is host-visible for zero-submit readback.
- `decode.c` — CPU two-stage top-300 (per-anchor max → top anchors → top anchor/class pairs) matching
  the in-graph ONNX TopK, producing `[x1,y1,x2,y2,score,cls]`.
- `yolo.c` — the C-ABI DLL (`yolo26.dll`): `yolo_open` / `yolo_infer` / `yolo_close` / `yolo_last_error`.
- `test_main.c` — CLI harness (`--bench`, `--timing`, `--input`, `--dump`, `--dump-tensor`).

### Shaders (`shader/`, GLSL 450, 256-thread workgroups)
`Im2col`, `CONV-GEMM-Q16` (16×32 fp16-block16 GEMM, channel-major NCHW output, fused bias+SiLU),
`CONV1x1-Q16` (transposed-A NCHW GEMM), `CONV-GEMM-FF32` (generic fp32 matmul), `DWConv`, `MaxPool`,
`Upsample2x`, `Elementwise` (add/sub/mul/sigmoid/scale/copy with broadcast periods), `Softmax`,
`Transpose2D`.

### Python (`python/yolo26vk`)
`Detector(model_dir, shader_dir=..., dll_path=...)` → `detect(image)` returns an `[N,6]` array in
original image coordinates. Letterbox + inverse box scaling in Python; tensor-in / detections-out via
`ctypes` (GIL released during the call).

## Build & run

```
set VULKAN_SDK=D:\VulkanSDK\1.4.350.0
make                       # builds bin/shader/*.spv, bin/test.exe, bin/yolo26.dll
bin\test.exe engine_model\s --shader bin\shader --bench 200
```

Regenerate an engine model from an ONNX export:

```
tools\export\.venv\Scripts\python.exe tools\export\parse_onnx.py model\small.onnx engine_model\s
```

Python:

```python
from yolo26vk import Detector
det = Detector("engine_model/s", shader_dir="bin/shader", dll_path="bin/yolo26.dll")
boxes = det.detect(pil_or_numpy_image)   # -> np.ndarray [N, 6] x1,y1,x2,y2,score,cls
```

## Notes & limitations

- `model/nano.pt` (YOLO26n), `model/small.pt` (custom YOLO26s, 4 classes) and `model/yolo26m.pt`
  (COCO) are the three exported models. `best.pt` (YOLO26l) is present but not exported.
- FP16 weights with FP32 accumulate: box error ≤ 0.8 px and class error ≤ 0.016 vs the FP32 ONNX
  reference. `CONV-GEMM-FF32` is available as a per-tensor FP32 escape hatch.
- The 3×3 path is im2col + GEMM (a working implicit-GEMM variant was measured ~5 % slower on this
  GPU and is not shipped). The dominant cost is the im2col/GEMM bandwidth; a larger N-tile was also
  measured slower due to occupancy, so the Pumice-verified 16×32 tile is kept.
- No NMS is needed — the YOLO26 one-to-one head is end-to-end.
- Single queue, serialized ops (global write→read barrier between every op), one `execute()` per frame.
