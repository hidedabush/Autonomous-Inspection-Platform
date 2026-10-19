"""Phase 1 sanity check: confirms PyTorch, CUDA and OpenCV are installed."""

import cv2
import torch


def main() -> None:
    print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"opencv {cv2.__version__}")


if __name__ == "__main__":
    main()
