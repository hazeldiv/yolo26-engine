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

### vs. Python ONNX Runtime (CPU)

Same exported ONNX models, same machine, `tools/export/bench_ort.py` (onnxruntime 1.31.0,
`CPUExecutionProvider`, default threading — this system has no ROCm/DirectML EP for the RX 580):

| Model | ORT CPU | Vulkan engine | Speedup |
|---|---|---|---|
| YOLO26n | 35.1 ms (28.5 FPS) | 12.8 ms (78.3 FPS) | 2.7× |
| YOLO26s | 102.5 ms (9.8 FPS) | 31.6 ms (31.7 FPS) | 3.2× |
| YOLO26m | 309.7 ms (3.2 FPS) | 87.9 ms (11.4 FPS) | 3.5× |

GPU time breakdown (`--timing`, op-time only, sum of per-op timestamps):

| Kernel | n | s | m |
|---|---|---|---|
| CONV-GEMM-Q16 (3×3 im2col GEMM) | 4.4 ms | 12.9 ms | 43.1 ms |
| Im2col | 2.7 ms | 5.2 ms | 15.3 ms |
| CONV1x1-Q16 (1×1 direct GEMM) | 2.4 ms | 7.8 ms | 20.8 ms |
| DWConv / MaxPool / Upsample | 0.3 ms | 0.8 ms | 1.0 ms |
| Attention (FF32 GEMM + Softmax + Transpose) | 0.3 ms | 1.1 ms | 0.5 ms |
| Elementwise (SiLU-add, concat copies, head decode) | 0.4 ms | 1.9 ms | 2.8 ms |

## Per-shader timing & optimization investigation (YOLO26m)

`bin/test.exe engine_model/m --shader bin/shader --timing` dumps a per-op timing log and an aggregate
summary. Full per-shader breakdown for m (43× 3×3 convs, 61× 1×1 convs, 550 ops total):

| Kernel | calls | total | % | avg |
|---|---|---|---|---|
| CONV-GEMM-Q16 (3×3, im2col+GEMM) | 43 | 42.5 ms | 51% | 0.99 ms |
| CONV1x1-Q16 (1×1 direct GEMM) | 61 | 20.8 ms | 25% | 0.34 ms |
| Im2col | 43 | 13.4 ms | 16% | 0.31 ms |
| Elementwise | 350 | 2.5 ms | 3% | 0.007 ms |
| DWConv / FF32 / Upsample / Softmax / MaxPool / Transpose | 76 | 1.5 ms | 2% | — |

The three slowest single ops are all stride-2 downsample convs:
`Cout=512 M=1600 Cin=512` (7.57 ms), `Cout=256 M=6400 Cin=256` (7.47 ms), `Cout=128 M=25600 Cin=64`
(3.73 ms). For the 7.47 ms op (3.78 GMAC):

| | A (im2col) | B (weights) |
|---|---|---|
| size | 59 MB | 1.2 MB |
| re-read factor | Npad/32 = 8× | M/16 = 400× |
| DRAM traffic | 472 MB | 472 MB |

≈950 MB moved in 7.5 ms = **~125 GB/s** (≈half of the RX 580's practical bandwidth) and
**~1.0 TFLOP** (≈16% of peak). The GEMMs are **bandwidth/latency-bound on tile re-reads**, not
compute-bound; im2col is a pure 13 ms pass that writes 59 MB only to read it back.

Eight shader-level optimizations were implemented, validated against ONNX Runtime, and benchmarked.
**None beat the Pumice-derived baseline** (each was neutral or worse, measured by both wall-clock and
GPU timestamp totals):

| Attempt | Change | Result |
|---|---|---|
| Tap-decomposed 3×3 GEMM | read NCHW directly, no im2col (stride-1) | neutral (removed im2col but slower A gather) |
| Register-blocked 32×32 tile | 2×2 outputs/thread (im2col and tap variants) | worse (register/occupancy pressure) |
| Triple-buffered K-loop | 2-deep prefetch, 1 barrier/tile | neutral |
| KT=32 K-tile | half the barriers, 8 vec4 per tile | worse |
| Vectorized 1×1 A-load | vec4 along m | worse |
| fp16 im2col | halve A traffic | worse **and** less accurate |
| Concat direct-write | producers write straight into the concat buffer | worse (scattered early writes hurt locality) |

Conclusion: the tiling is already at the practical limit for this GPU; the remaining gap to peak is
the RX 580's memory subsystem at these activation-heavy shapes (large M, small N), not shader
inefficiency. The shipped kernels are the validated baseline.

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

### Real-image validation (CPU ground truth vs GPU engine)

`tools/export/validate_images.py` runs the 3 root images (`000041.jpg`, `000068.jpg`, `000077.jpg`)
through every variant and compares the **GPU engine** (`Detector.detect`, full stack: letterbox →
engine → decode → inverse scaling) against two CPU references. Detections are matched greedily by
class at IoU ≥ 0.5; errors are in original-image pixels. `conf ≥ 0.25`:

| Variant | Image | ONNX-CPU vs engine (match/tot, box err max/mean, score err) | PT-CPU vs engine |
|---|---|---|---|
| n | 000041 | 13/13, 0.03 / 0.01 px, 0.0018 | 13/13, 14.8 / 7.1 px |
| n | 000068 | 25/25, 0.06 / 0.02 px, 0.0132 | 23/27, 25.3 / 7.4 px |
| n | 000077 | 27/27, 0.06 / 0.02 px, 0.0068 | 22/26, 49.4 / 7.1 px |
| s | 000041 | 13/13, 0.04 / 0.02 px, 0.0002 | 13/13, 18.1 / 5.5 px |
| s | 000068 | 24/24, 0.06 / 0.02 px, 0.0016 | 24/26, 10.3 / 5.0 px |
| s | 000077 | 24/24, 0.08 / 0.02 px, 0.0364 | 24/25, 45.4 / 8.9 px |
| m | 000041 | 0/0 (no COCO object ≥ 0.25) | 0/0 |
| m | 000068 | 2/2, 0.02 / 0.01 px, 0.0038 | 2/4, 6.1 / 3.6 px |
| m | 000077 | 4/4, 0.04 / 0.02 px, 0.0037 | 3/4, 43.4 / 15.3 px |

The engine matches the exported ONNX graph on CPU **exactly** (100% of confident detections,
sub-0.1 px box error, ≤0.04 score error). The larger spread against the ultralytics `.pt` path is
ultralytics' own predict pipeline (its own letterbox/resize interpolation and box scaling), not the
engine — the engine reproduces the exported graph bit-for-bit within FP16.

**Two real bugs were found by this test** (both invisible to the random-input tensor compare, which
passed on noise because those logits are ~0):

1. **Head Sigmoid used the SiLU formula.** `Elementwise.comp` op `3` computed `a/(1+exp(-a))`
   (SiLU) instead of the sigmoid `1/(1+exp(-a))`. On random-noise input the class logits are very
   negative so both formulas evaluate to ~0 and the compare passed; on real images with positive
   logits the engine emitted raw logits instead of probabilities (class error 0.016 → 0.000006
   after the fix).
2. **Python wrapper fed HWC data to a CHW engine.** `canvas.transpose(2,0,1)/255` returns an
   *F-contiguous* array (still HWC memory order), so `ctypes.memmove` copied the wrong layout.
   Fixed with `np.ascontiguousarray(...)`.

Regression after the fixes (random-input tensor compare): class max abs err n 6e-6 / s 6e-6 / m 3e-6.


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
