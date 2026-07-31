"""Benchmarks two C++ Monte Carlo kernels — single-threaded and multi-threaded
— against the pure Python/NumPy version from Part A, at increasing path
counts, and plots the results.

Build the extension first:
    cd perf && python setup.py build_ext --inplace && cd ..

Then run from the repo root:
    python -m perf.benchmark
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt

from pricing.black_scholes import bs_price
from pricing.monte_carlo import naive_mc_price

FIG_DIR = Path(__file__).parent / "figures"

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
PATH_COUNTS = [10_000, 100_000, 1_000_000, 5_000_000, 20_000_000]


def _time_python(n_paths: int) -> tuple[float, float]:
    start = time.perf_counter()
    result = naive_mc_price(S, K, T, r, sigma, n_paths, "call", seed=1)
    return time.perf_counter() - start, result.price


def _time_cpp_single(n_paths: int) -> tuple[float, float]:
    from perf import mc_pricer_cpp

    start = time.perf_counter()
    result = mc_pricer_cpp.mc_price(S, K, T, r, sigma, n_paths, "call", 1)
    return time.perf_counter() - start, result.price


def _time_cpp_parallel(n_paths: int) -> tuple[float, float]:
    from perf import mc_pricer_cpp

    start = time.perf_counter()
    result = mc_pricer_cpp.mc_price_parallel(S, K, T, r, sigma, n_paths, "call", 1, 0)
    return time.perf_counter() - start, result.price


def run_benchmark() -> list[dict]:
    bs_ref = bs_price(S, K, T, r, sigma, "call")
    rows = []
    for n in PATH_COUNTS:
        py_time, py_price = _time_python(n)
        cpp_time, cpp_price = _time_cpp_single(n)
        cpp_par_time, cpp_par_price = _time_cpp_parallel(n)
        rows.append({
            "n_paths": n,
            "python_seconds": py_time, "cpp_seconds": cpp_time, "cpp_parallel_seconds": cpp_par_time,
            "speedup_vs_naive_cpp": py_time / cpp_time,
            "speedup_vs_parallel_cpp": py_time / cpp_par_time,
            "python_price": py_price, "cpp_price": cpp_price, "cpp_parallel_price": cpp_par_price, "bs_ref": bs_ref,
        })
    return rows


def plot_benchmark(rows: list[dict]):
    n_paths = [r["n_paths"] for r in rows]
    py_times = [r["python_seconds"] for r in rows]
    cpp_times = [r["cpp_seconds"] for r in rows]
    cpp_par_times = [r["cpp_parallel_seconds"] for r in rows]
    speedups = [r["speedup_vs_parallel_cpp"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(n_paths, py_times, marker="o", label="Python / NumPy (vectorized)")
    ax1.plot(n_paths, cpp_times, marker="o", label="C++ single-threaded")
    ax1.plot(n_paths, cpp_par_times, marker="o", label="C++ multi-threaded (std::thread)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of simulated paths")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Monte Carlo pricer: wall-clock time")
    ax1.legend(fontsize=8)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Parity with Python")
    ax2.plot(n_paths, speedups, marker="o", color="darkorange", label="Multi-threaded C++ speedup")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of simulated paths")
    ax2.set_ylabel("Speedup (Python time / C++ time)")
    ax2.set_title("Multi-threaded C++ vs. Python/NumPy")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "cpp_speedup.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    rows = run_benchmark()
    for r in rows:
        print(
            f"n={r['n_paths']:>10,}  python={r['python_seconds']:.4f}s  "
            f"cpp_single={r['cpp_seconds']:.4f}s (speedup={r['speedup_vs_naive_cpp']:.2f}x)  "
            f"cpp_parallel={r['cpp_parallel_seconds']:.4f}s (speedup={r['speedup_vs_parallel_cpp']:.2f}x)  "
            f"prices: py={r['python_price']:.4f} cpp={r['cpp_price']:.4f} cpp_par={r['cpp_parallel_price']:.4f} bs={r['bs_ref']:.4f}"
        )
    plot_benchmark(rows)
    print("\nSaved figure to", FIG_DIR)
