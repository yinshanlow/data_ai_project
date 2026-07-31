// Monte Carlo European option pricer — the same algorithm as
// pricing/monte_carlo.py::naive_mc_price, rewritten in C++ and exposed to
// Python via pybind11.
//
// Two versions are exposed on purpose, not just one: a plain single-threaded
// C++ loop (mc_price), and a multi-threaded version using std::thread
// (mc_price_parallel, no external dependency — Homebrew's libomp wasn't
// available to install without changing system directory permissions this
// build didn't have, so this uses the C++ standard library's own threading
// instead of OpenMP). See perf/README.md for why both exist: the naive
// single-threaded version is NOT automatically faster than NumPy's own
// vectorized (SIMD, C-backed) random number generation and array ops — a
// real, worth-reporting finding, not the result a "C++ is always faster"
// assumption would predict. The multi-threaded version is what actually
// delivers a genuine, provable speedup, and it's the more honest place to
// make the "I have real C++ performance-engineering experience" case.
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

struct MCResult {
    double price;
    double std_error;
};

struct PartialSums {
    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;
};

static PartialSums simulate_chunk(
    double S, double K, double T, double r, double sigma,
    long n_paths, const std::string &option_type, unsigned int seed
) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double drift = (r - 0.5 * sigma * sigma) * T;
    const double vol_sqrt_t = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);

    PartialSums sums;
    for (long i = 0; i < n_paths; ++i) {
        double z = normal(rng);
        double S_T = S * std::exp(drift + vol_sqrt_t * z);
        double payoff = (option_type == "call")
            ? std::max(S_T - K, 0.0)
            : std::max(K - S_T, 0.0);
        double discounted = discount * payoff;
        sums.sum_payoff += discounted;
        sums.sum_payoff_sq += discounted * discounted;
    }
    return sums;
}

static MCResult combine(const std::vector<PartialSums> &chunks, long n_paths) {
    double sum_payoff = 0.0, sum_payoff_sq = 0.0;
    for (const auto &c : chunks) {
        sum_payoff += c.sum_payoff;
        sum_payoff_sq += c.sum_payoff_sq;
    }
    double mean = sum_payoff / static_cast<double>(n_paths);
    double variance = sum_payoff_sq / static_cast<double>(n_paths) - mean * mean;
    variance = std::max(variance, 0.0); // guards against tiny negative values from floating-point cancellation
    double std_error = std::sqrt(variance / static_cast<double>(n_paths));
    return {mean, std_error};
}

static void validate_option_type(const std::string &option_type) {
    if (option_type != "call" && option_type != "put") {
        throw std::invalid_argument("option_type must be 'call' or 'put'");
    }
}

MCResult mc_price_cpp(
    double S, double K, double T, double r, double sigma,
    long n_paths, const std::string &option_type, unsigned int seed
) {
    validate_option_type(option_type);
    std::vector<PartialSums> chunks = {simulate_chunk(S, K, T, r, sigma, n_paths, option_type, seed)};
    return combine(chunks, n_paths);
}

MCResult mc_price_parallel_cpp(
    double S, double K, double T, double r, double sigma,
    long n_paths, const std::string &option_type, unsigned int seed, unsigned int n_threads
) {
    validate_option_type(option_type);
    if (n_threads == 0) {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    n_threads = static_cast<unsigned int>(std::min<long>(n_threads, std::max<long>(n_paths, 1)));

    std::vector<std::thread> workers;
    std::vector<PartialSums> results(n_threads);
    long base_chunk = n_paths / n_threads;
    long remainder = n_paths % n_threads;

    for (unsigned int t = 0; t < n_threads; ++t) {
        long this_chunk = base_chunk + (static_cast<long>(t) < remainder ? 1 : 0);
        unsigned int thread_seed = seed + t * 7919u; // distinct, well-spread seed per thread
        workers.emplace_back([&, t, this_chunk, thread_seed]() {
            results[t] = simulate_chunk(S, K, T, r, sigma, this_chunk, option_type, thread_seed);
        });
    }
    for (auto &w : workers) {
        w.join();
    }

    return combine(results, n_paths);
}

PYBIND11_MODULE(mc_pricer_cpp, m) {
    m.doc() = "C++ Monte Carlo European option pricer (pybind11 binding), single- and multi-threaded";

    py::class_<MCResult>(m, "MCResult")
        .def_readonly("price", &MCResult::price)
        .def_readonly("std_error", &MCResult::std_error);

    m.def(
        "mc_price", &mc_price_cpp,
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
        py::arg("n_paths"), py::arg("option_type") = "call", py::arg("seed") = 42,
        "Single-threaded Monte Carlo European option price, computed entirely in C++."
    );

    m.def(
        "mc_price_parallel", &mc_price_parallel_cpp,
        py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
        py::arg("n_paths"), py::arg("option_type") = "call", py::arg("seed") = 42, py::arg("n_threads") = 0,
        py::call_guard<py::gil_scoped_release>(),
        "Multi-threaded (std::thread) Monte Carlo European option price. n_threads=0 uses "
        "std::thread::hardware_concurrency(). GIL is released for the duration of the call."
    );
}
