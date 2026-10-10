# yolo26-engine

A from-scratch Vulkan compute inference engine for **YOLO26** (n / s / m), written in C + GLSL.
It loads an `.onnx` file directly, runs static shape inference, lowers the graph to engine ops and
repacks the weights in memory — no export or conversion step. Fixed 640×640 input, FP16 weights with
FP32 accumulation.

## Requirements

- Windows, MinGW gcc, GNU make
- Vulkan SDK (`VULKAN_SDK` set; glslangValidator on PATH under it)
- A Vulkan 1.1 device (developed on an RX 580)

## Build

```
set VULKAN_SDK=D:\VulkanSDK\1.4.350.0
make
```

Builds `bin/shader/*.spv`, `bin/test.exe` and `bin/yolo26.dll`.

## Run

```
bin\test.exe model\small.onnx --shader bin\shader --bench 200
```

Options: `--bench N`, `--timing`, `--input <raw.f32>`, `--dump <prefix>`, `--dump-tensor <name>`,
`--dump-plan <file>`, `--meminfo`.

## Python

```
tools\export\.venv\Scripts\python.exe python\bench.py --runs 100
```

```python
from yolo26vk import Detector

det = Detector("model/small.onnx", shader_dir="bin/shader", dll_path="bin/yolo26.dll")
boxes = det.detect(image)   # np.ndarray [N, 6] -> x1, y1, x2, y2, score, cls
```

## Performance (RX 580, 640×640, batch 1)

| Model | Per-frame | FPS |
|---|---|---|
| YOLO26n | 11.3 ms | 88.8 |
| YOLO26s | 24.4 ms | 41.0 |
| YOLO26m | 59.6 ms | 16.8 |

## Layout

- `src/onnx.c` — ONNX protobuf reader
- `src/compile.c` — shape inference, graph lowering, weight repacking
- `src/graph.c` — Vulkan resources and execution
- `src/yolo.c` — C-ABI DLL (`yolo_open` / `yolo_infer` / `yolo_close`)
- `src/decode.c` — CPU top-300 decode
- `shader/` — GLSL compute kernels
- `python/yolo26vk` — ctypes wrapper
- `tools/export/` — validation and benchmarking scripts

See `SUMMARY.md` for the full write-up.
