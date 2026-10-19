# Development setup

## Everyone
- Git, GitHub account added to the org/repo, Jira access, shared Google Drive access.

## C++ / CUDA (Linux recommended; Windows works with Visual Studio 2022)
1. Install a C++17 compiler and CMake 3.24+.
2. Install the CUDA Toolkit that matches your GPU (`nvidia-smi` shows the max supported version).
3. Optional for now: OpenCV 4.x (needed from Phase 2).
4. Build: see README.

Record your setup in the table so others can debug against it:

| Member | OS | GPU | Driver | CUDA | Compiler |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Python / ML
```bash
cd ml && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Firmware
- VS Code + PlatformIO extension, then open `firmware/esp32`.
