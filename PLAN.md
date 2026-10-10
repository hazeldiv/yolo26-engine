# YOLO26 Vulkan Inference Engine — Implementation Plan

## Context

YOLO26 doesn't run efficiently on the RX 580 with existing runtimes (ONNX Runtime CPU: 39/87/220 ms for n/s/m at 640). This project builds a from-scratch Vulkan compute inference engine in C + GLSL for YOLO26 **n/s/m**, based on the proven architecture of the Pumice LLM engine (`D:\coding\python\vk_engine`) — its Vulkan core, `operation` dispatch API, fp16 block-transposed weight layout, and RX 580 (GCN4, wave64)-tuned 16×16/256-thread GEMM tiling.

Decisions already made with the user:
- **Python library interface** — a C-ABI DLL (`yolo26.dll`) called from Python via `ctypes` (no subprocess). numpy in, detections out.
- **FP16 weights, fp32 accumulate** for v1 (m ≈ 41 MB — VRAM is never the constraint here).
- **Fixed 640×640 input**.
- Test model: `model/small.pt` — a **custom-trained YOLO26s** (verified: `yolo26s.yaml`, Detect/C3k2/SPPF/C2PSA/DWConv). Engine reads all dims at runtime from the export (n/s/m same op set, different depth/width).
- Toolchain (verified present): MinGW gcc 11.2, Vulkan SDK 1.4.350 (`D:\VulkanSDK\1.4.350.0`), GNU make under cmd.exe.

Performance targets (v1): **n ~8-12 ms, s ~15-30 ms, m ~40-60 ms** per 640 frame.

YOLO26 architecture (from `yolo26.yaml`): Conv 3×3/1×1, C3k2 blocks, SPPF (3× 5×5 maxpool + concat), C2PSA (small MHSA), PAN-FPN head (Upsample nearest 2×, Concat, downsample convs), dual Detect head with **no DFL** (`reg_max=1`), end-to-end one-to-one branch outputting `(N, 300, 6)` = `[x1,y1,x2,y2,score,cls]` — no NMS. Scales: n [0.50, 0.25, 1024], s [0.50, 0.50, 1024], m [0.50, 1.00, 512].

## Key design decisions

1. **Export via ONNX** (not nn.Module walking): `.pt` → `export(format="onnx", imgsz=640, nms=False, simplify=True, opset≥14)` → **onnxsim** (folds Shape/Gather/Constant chains) → Python parser → `graph.json` + `weights.bin`. ONNX is a frozen flat graph, immune to ultralytics version churn; torch is a one-time build-tool dependency only (uv venv; system Python has no torch).
2. **Conv = im2col + adapted Pumice GEMM template** ([GEMM-ADD2-Q16.comp](../python/vk_engine/shader/Utility/Q16/GEMM-ADD2-Q16.comp)) for 3×3 (tap-major k; one shader covers s1/s2 via push constant). **1×1 convs skip im2col** — NCHW `[C][H][W]` is already a `[H·W][C]` GEMM A-operand. **DWConv** gets a direct small shader. Implicit-GEMM (gather fused into A-tile load) is the planned v1.1 optimization if the timing log shows im2col bandwidth matters.
3. **SiLU + bias fused into the GEMM epilogue** (Pumice fused-epilogue precedent), gated by a push-constant flag with clamped N-writes.
4. **Concat/Channel-split are free metadata**: producer shaders take a `colOff` write offset; consumers take `Aoff` read offset + `Astride` (Pumice's LinearProj kOff/vOff routing precedent). Activation buffers: one named VRAM buffer per tensor, allocated by liveness analysis over the op-ordered DAG (sound because [dispatch.c](../python/vk_engine/src/dispatch.c) serializes ops with a global WRITE→READ barrier between every pair).
5. **C2PSA needs no special-casing**: generic fp32×fp32 MatMul (≤400×400×d) + Transpose-copy + Softmax + elementwise covers whatever attention soup the ONNX graph emits — this generic set also de-risks the head decode region.
6. **Detect head**: everything up to (excluding) the in-graph TopK runs on GPU; exporter DCEs the one-to-many branch (~half the head FLOPs). CPU (in the DLL) does top-300 + gather (~50 µs).
7. **~120-250 ops/frame, one `execute()` per frame**, re-recorded each frame (cached pipelines/descriptors make this tens of µs); frame GPU time 15-60 ms is far under the 8 s TDR limit.
8. **Python does letterbox + inverse box scaling**; the DLL is a pure tensor-in/tensor-out contract (640×640×3 CHW fp32 RGB [0,1]). CPU top-k stays in C (touches raw readback before any Python object exists).

## Repo layout (`d:\coding\yolo26-engine`)

```
Makefile                    (adapted from Pumice: DLL + test-exe targets, no node/tokenizers)
include/                    buffer.h device.h descriptor.h dispatch.h fence.h pipeline.h session.h json.h (VERBATIM)
                            graph.h (NEW: graph.json model + op compiler), yolo.h (NEW: DLL C API), validation.h (NEW)
src/                        buffer.c device.c descriptor.c dispatch.c fence.c pipeline.c session.c json.c (VERBATIM)
                            graph.c (NEW: graph loader + liveness + op compiler → operation[])
                            weights.c (NEW: plain reads of weights.bin — all layout done by exporter)
                            decode.c (NEW: readback + CPU top-300 + gather)
                            yolo.c (NEW: DLL entry points, per-frame driver)
                            validation.c test_main.c (NEW: CPU-ref harness + CLI)
shader/
  Utility/Im2col.comp                 (NEW)
  Conv/CONV-GEMM-Q16.comp             (ADAPTED from GEMM-ADD2-Q16.comp)
  Conv/CONV-GEMM-FF32.comp            (NEW: generic fp32×fp32 MatMul)
  Conv/DWConv.comp  MaxPool.comp  Upsample2x.comp  (NEW)
  Elementwise/Add.comp Mul.comp Sigmoid.comp Mul-Sigmoid.comp ScaleShift.comp  (NEW)
  Attention/Softmax.comp              (ADAPTED from Att-Softmax)
  Attention/Transpose.comp            (NEW: generic copy-transpose)
tools/export/               (uv venv: pinned torch+ultralytics+onnx+onnxsim)
  export_yolo26.py parse_onnx.py dump_reference.py
python/yolo26vk/            __init__.py _core.py (ctypes wrapper: Detector class)
engine_model/<name>/        export artifacts: graph.json + weights.bin
bin/                        shader/*.spv, yolo26.dll, test.exe
model/                      small.pt (existing custom YOLO26s); n/m .pt added at M6 (ultralytics auto-download or user-provided)
```

Preserve Pumice Makefile quirks: cmd.exe recipes with backslash paths, recursive rwildcard → flat `bin\shader\*.spv` via `glslangValidator -V --target-env vulkan1.1`, no spaces in shader folder names, **no header dep tracking — `make clean` after any header edit**.

## Export pipeline (tools/export)

1. `export_yolo26.py`: `YOLO(model).export(format="onnx", imgsz=640, nms=False, dynamic=False, simplify=True)` → onnxsim pass.
2. `parse_onnx.py`:
   - Build SSA tensor DAG from the node list.
   - **DCE from the cut point**: locate the TopK on the one-to-one path; the scores tensor (~[1,8400]) and the gather source (~[1,8400,6]) become engine readback tensors. Drop the one-to-many branch and everything feeding only dropped outputs.
   - Translate mechanically: Conv → im2col+conv or direct conv (SiLU fusion when `Conv → Mul(x, Sigmoid(x))` pattern appears); MaxPool; Resize(nearest 2×) → upsample; Concat/Split → routing offsets; Add/Mul/Sigmoid/Div → elementwise; Reshape/Squeeze/Unsqueeze → metadata (assert layout-preserving, else Transpose-copy); Transpose → copy shader; MatMul → GEMM-FF32; Softmax → softmax.
   - **Unmapped node → hard error naming the node** (never guess).
   - **Padding rules** (the GEMM template has no bounds checks): weights padded to `Kpad = ceil16(K)` (zero tail) and `Npad32 = ceil32(N)`; im2col zero-fills K-tail; clamped epilogue writes only `n < Ntrue` at `colOff`; `M = H·W` is always a 16-multiple (6400/1600/400/25600/102400 — verified).
   - Emit `graph.json` (op list with per-op push-constant fields, tensor table, weight table with binOffsets, `post` recipe) + `weights.bin` (flat: fp16-block16 GEMM weights — exporter reimplements `transpose_block16` in numpy; fp16-raw-c-major DWConv weights `[C][k·k]`; fp32-raw biases/constants).
3. `dump_reference.py`: run the `.pt` model on test images with forward hooks; dump ~15 named intermediate activations + final detections as `.npy`/JSON.

Fallback if the head region resists translation: cut right after the one2one head convs and implement the documented ultralytics decode (sigmoid cls, dist2bbox, strides 8/16/32, top-300) in `decode.c`, validated end-to-end.

## Shader specs (all: GLSL 450, set=0 bindings 0..n-1 STORAGE_BUFFER, push constants ≤16 ints, 256-thread workgroups = 4× wave64 — the Pumice-verified GCN4 sweet spot, no float atomics anywhere, ceil() dispatch grids, no barrier after any conditional return)

- **Im2col.comp** — [C,H,W] fp32 → [M=Hout·Wout][Kpad] with k tap-major (`k = tap·Cin + c`) for contiguous vec4 channel reads; zero-fills K-tail + out-of-border taps. 256 threads, one vec4 of one output row each. Push: inW, inH, Cin, Aoff, Kpad, stride, pad, outW, outH.
- **CONV-GEMM-Q16.comp** — adapted template: bindings 0=A, 1=B(fp16 block16), 2=C, 3=bias. Push: M, K, Npad32, Cstride, Ntrue, colOff, Aoff, Astride, flags(silu) — 9 ints. Epilogue: `v = acc + bias[n]; if(silu) v = v·sigmoid(v); if(n < Ntrue) C[m·Cstride + colOff + n] = v`. Dispatch `x = Npad32/32, y = M/16`. Astride decouples A row stride from K (1×1 direct view, split reads, compact-concat inputs).
- **CONV-GEMM-FF32.comp** — same skeleton, B fp32 row-major (activation×activation MatMuls; ≤400×400×d).
- **DWConv.comp** — one thread per (pixel, channel); k×k taps + fp16 `[C][k·k]` weight + bias, optional SiLU, colOff routing, zero-pad border.
- **MaxPool.comp** — one thread per (pixel, channel), k×k window (SPPF 5×5 s1 p2), colOff routing.
- **Upsample2x.comp** — nearest 2× gather, colOff routing; Resize attrs verified against the simplified ONNX at M3.
- **Elementwise/** — Add, Mul, Sigmoid, Mul-Sigmoid, ScaleShift (per-channel const broadcast); one thread per element (or vec4); colOff/Aoff routing. Covers residual adds, stride multiplies, and head decode arithmetic.
- **Softmax.comp** — one 256-thread workgroup per row, two-pass max/exp-sum in shared.
- **Transpose.comp** — generic 2D copy-transpose via shared-memory tile.

## Engine core (C)

- **Verbatim copies**: buffer/device/descriptor/fence/pipeline/session/json/dispatch .c/.h from Pumice (~1000 lines total; `execute()` used as-is with its barrier serialization + timestamp timing).
- **graph.c**: parse graph.json → create weight buffers (VRAM, one batched `createTransferAndCopy`) → create activation buffers from sizeElems (zero-init once via staging) → liveness allocation (reuse buffer when `producerIndex(new) > lastConsumerIndex(old)`) → compile each op into `operation{}` (shader name, bindings in order, push constants, dispatch XYZ). Buffer counts (~150-250 activation + ~100-150 weight) fit the descriptor cache (1024) / pipeline cache (4096) / timestamp pool (8192).
- **decode.c**: `readBuffer` the two post tensors → CPU partial-sort top-300 → gather rows → `yolo_detection[300]`.
- **yolo.c**: per frame: upload input (writeStaging + transfer submit) → `execute()` → readback → decode.
- **test_main.c**: `test.exe <model_dir> [--validate] [--image x] [--dump-tensor tN] [--timing] [--bench 100]`.

## DLL C API + Python wrapper

```c
typedef struct { float x1, y1, x2, y2, score; int cls; } yolo_detection;
int  yolo_open(const char* model_dir, const char* shader_dir, yolo_model** out);
int  yolo_infer(yolo_model* m, const float* rgb /*3*640*640 CHW fp32 RGB [0,1], letterboxed*/,
               yolo_detection* dets, int max_dets, int* n_dets);   /* ≤300, blocking */
void yolo_close(yolo_model* m);
const char* yolo_last_error(void);
```

Python (`python/yolo26vk`):
```python
class Detector:
    def __init__(self, model_dir, shader_dir=None): ...  # CDLL("yolo26.dll"), yolo_open
    def detect(self, image) -> np.ndarray:
        # PIL/numpy letterbox to 640×640 (record pad/gain); float32 CHW /255
        # yolo_infer (ctypes releases the GIL during the call)
        # returns [n,6] x1,y1,x2,y2,score,cls in ORIGINAL image coords (subtract pad, divide gain)
```
A single handle is single-queue serialized (document; internal mutex if trivial).

## VRAM budget (m model, 640, fp32 activations / fp16 weights)

| Item | Size |
|---|---|
| Weights fp16 (~20M params) + biases | ~41 MB |
| Input 640×640×3 fp32 | 5 MB |
| Activation arena (liveness-packed) | ~120-180 MB |
| Shared im2col scratch (max: P1 `[102400][576]`) | 236 MB |
| **Total** | **~450 MB** (vs ~7936 MB usable) |

s tops out at a 118 MB scratch. No constraint.

## Validation (3 layers)

1. **Per-op CPU refs** (`--validate`): random fp32 inputs through the real dispatch path vs naive C references (conv w/ padding+stride, dwconv, maxpool, nearest-upsample, softmax, matmul, elementwise, block16 roundtrip). Must-pass edge cases: stem 3×3 s2 (Cin=3), 1×1 direct with K%16≠0 (Cin=24 → Kpad 32), colOff≠0, Aoff≠0, Cout=16<32 tile (n-scale), 400×400 attention shapes.
2. **Graph tensor compare**: `dump_reference.py` activations vs `test.exe --dump-tensor` readback; per-tensor max-abs tolerance assertions. Localizes any graph bug to one op.
3. **End-to-end**: engine detections vs ultralytics `predict()` on test images — IoU > 0.95, score Δ < 0.02, identical classes. fp16 expectations: ~2-4e-2 worst-case on deep features, boxes within ~1-2 px; escape hatch = keep offending weight tensor fp32 via the GEMM-FF32 path.

## Milestones

- **M0 — Vulkan smoke test** (half day): copy verbatim files + Makefile; run one Pumice GEMM validation through the new build. Confirms toolchain end-to-end.
- **M1 — Conv core validated** (2-3 days): Im2col + CONV-GEMM-Q16 + all Layer-1 edge cases; timing on m-scale shapes (102400×576×256). Exit: fp16-tolerance errors, conv < 3 ms at s-scale, im2col-vs-GEMM split known.
- **M2 — Support ops validated** (1-2 days): DWConv, MaxPool, Upsample, elementwise, Softmax, GEMM-FF32, Transpose. Exit: `--validate` green.
- **M3 — Export + full graph** (3-5 days, highest variance): uv venv, export small.pt, hand-inspect the simplified ONNX once, finish parse_onnx.py, run the graph, Layer-2 tensor compare green.
- **M4 — Decode + end-to-end** (1-2 days): decode.c per post recipe; Layer-3 detection compare.
- **M5 — DLL + Python** (1 day): yolo26.dll target, ctypes wrapper, `Detector.detect()` matches M4; 100-frame bench with `--timing` breakdown.
- **M6 — n/m + tuning** (1-2 days): export yolo26n.pt / yolo26m.pt (pretrained COCO download or user-provided), verify zero hardcoded dims in C, full validation + timing on all three; optional tuning (workgroup variants, implicit-GEMM conv).

## Risk register (highest first)

1. **ONNX head/decode region drift** (TopK pattern, Resize attrs) — de-risk: onnxsim + generic op set + hard-fail errors naming nodes; fallback = earlier cut + known ultralytics decode in C.
2. **GEMM template invariants** (no bounds checks; K%16, N%32, M%16) — de-risk: exporter-side padding rules + clamped epilogue + Astride/Aoff; every odd-shape case in Layer 1.
3. **im2col bandwidth on m** (236 MB scratch, 9× traffic) — de-risk: measured at M1 before anything else; implicit-GEMM is the planned fix; the actual test model (s) tops at 118 MB.
4. **fp16 accuracy on the custom checkpoint** — de-risk: Layer-2 per-tensor attribution; per-tensor fp32 fallback.
5. **ultralytics/torch venv drift** — de-risk: pinned requirements; runtime has zero torch dependency (artifacts frozen).

## Reference files to reuse

- [GEMM-ADD2-Q16.comp](../python/vk_engine/shader/Utility/Q16/GEMM-ADD2-Q16.comp) — the tiling template CONV-GEMM-Q16 adapts (63 lines: ping-pong shared tiles, fp16 block16 B loads, vec4 dot FMAs)
- [dispatch.c](../python/vk_engine/src/dispatch.c) + [dispatch.h](../python/vk_engine/include/dispatch.h) — execute/recordFrame/barrier machinery, used verbatim; the op-serialization contract liveness allocation depends on
- [data.c `transpose_block16`](../python/vk_engine/src/data.c) — weight layout the exporter reimplements in numpy
- [buffer.c](../python/vk_engine/src/buffer.c), [device.c](../python/vk_engine/src/device.c), [session.c](../python/vk_engine/src/session.c), [pipeline.c](../python/vk_engine/src/pipeline.c), [descriptor.c](../python/vk_engine/src/descriptor.c) — verbatim Vulkan core
- [Makefile](../python/vk_engine/Makefile) — build system to adapt
- `vk_engine/docs/PUMICE-SUMMARY.md` §9 gotchas — GCN4 rules baked into the shader checklist (partial-workgroup barriers, no float atomicAdd, ceil dispatches, make clean after header edits, TDR slicing)
