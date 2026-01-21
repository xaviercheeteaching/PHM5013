import time
import numpy as np

def hello_world():
    """Basic greeting"""
    print("Hello World from the supercomputer!")
    print("1 + 1 =", 1 + 1)
    print("-" * 50)

def cpu_computation():
    """Single-core CPU computation"""
    print("Running CPU computation...")
    start = time.time()

    # Simple matrix multiplication
    size = 5000
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)

    elapsed = time.time() - start
    print(f"CPU computation completed in {elapsed:.2f} seconds")
    print(f"Result shape: {c.shape}")
    print("-" * 50)

def multicore_cpu_computation(n_cores=4):
    """Multi-core CPU computation using NumPy threading"""
    import os
    os.environ['OMP_NUM_THREADS'] = str(n_cores)

    print(f"Running multi-core CPU computation with {n_cores} cores...")
    start = time.time()

    size = 5000
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)

    elapsed = time.time() - start
    print(f"Multi-core computation completed in {elapsed:.2f} seconds")
    print(f"Cores used: {n_cores}")
    print("-" * 50)

def gpu_computation():
    """GPU computation if available"""
    try:
        import cupy as cp

        print("Running GPU computation...")
        start = time.time()

        size = 5000
        a = cp.random.rand(size, size)
        b = cp.random.rand(size, size)
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()

        elapsed = time.time() - start
        print(f"GPU computation completed in {elapsed:.2f} seconds")
        print(f"Result shape: {c.shape}")
        print("GPU device:", cp.cuda.Device())

    except ImportError:
        print("CuPy not available - GPU computation skipped")
    except Exception as e:
        print(f"GPU error: {e}")

    print("-" * 50)

if __name__ == "__main__":
    import sys

    hello_world()

    # Check command line argument for mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    if mode == "cpu":
        cpu_computation()
    elif mode == "multicore":
        n_cores = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        multicore_cpu_computation(n_cores)
    elif mode == "gpu":
        gpu_computation()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python hello_compute.py [cpu|multicore|gpu] [n_cores]")
