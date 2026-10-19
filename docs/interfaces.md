# Interfaces between teams (DRAFT – agree by end of week 5)

Changes here need sign-off from every team listed on the interface.

## 1. Hardware -> C++: serial protocol (Hardware, C++/Graphics)

- Link: USB serial, 115200 baud, newline-terminated ASCII.
- Board -> PC: `TRIG <part_id>` when a part is in position; `ACK <cmd>` after a command; `ERR <msg>`.
- PC -> Board: `PASS`, `REJECT`, `HOME`, `PING`.
- Timeout: PC waits up to ___ ms for `ACK`.

## 2. C++ -> CUDA: image buffer (C++/Graphics, GPU/CUDA)

- `aip::Image`: 8-bit, interleaved, row-major, RGB order, no row padding.
- Resolution: ___ x ___ (to be fixed after camera choice).

## 3. CUDA -> ML: model input (GPU/CUDA, ML/CV)

- Input size: ___ x ___, channels: ___, normalization: ___.
- Layout: NCHW float32.

## 4. ML -> C++: model output (ML/CV, C++/Graphics)

- Verdict (PASS/FAIL), defect class (string from a fixed list), confidence 0–1, optional box or anomaly heatmap.
- Model file: ONNX, opset ___.

## 5. Logging schema (all)

| Column | Type | Notes |
| --- | --- | --- |
| id | integer | auto |
| timestamp | text | ISO 8601 |
| part_id | text | from trigger |
| verdict | text | PASS / FAIL |
| defect_class | text | empty if PASS |
| confidence | real | |
| latency_ms | real | trigger to verdict |
| backend | text | cpu / cuda |
| image_ref | text | path or Drive id |
