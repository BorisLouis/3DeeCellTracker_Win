"""Smoke-test the local notebook environment for 3DeeCellTracker."""

from __future__ import annotations

import importlib
import sys


MODULES = [
    "tensorflow",
    "csbdeep",
    "stardist",
    "matplotlib",
    "notebook",
    "numpy",
    "sklearn",
    "skimage",
    "h5py",
    "tifffile",
    "tqdm",
    "scipy",
    "PIL",
    "CellTracker",
]


def main() -> None:
    print("Python executable:", sys.executable)

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "import ok")
        print(f"{module_name}: {version}")

    import tensorflow as tf

    print("TensorFlow GPUs:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
