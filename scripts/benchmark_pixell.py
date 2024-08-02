import subprocess
import multiprocessing
import sys
from pathlib import Path


def main():
    max_threads = multiprocessing.cpu_count()
    assert max_threads >= 1

    def run_benchmark(nthreads):
        subprocess.call(
            [sys.executable, Path(__file__).parent / "benchmark_pixell_runner.py"],
            env={"OMP_NUM_THREADS": str(nthreads)},
        )

    print("Single threaded alm test:")
    run_benchmark(1)

    print(f"Multi-threaded alm test with {max_threads} threads:")
    run_benchmark(max_threads)
