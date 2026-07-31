"""Builds the C++ Monte Carlo pricer as a Python extension module.

Run with (from the perf/ directory):
    python setup.py build_ext --inplace

This uses pybind11's setuptools helper rather than CMake — cmake isn't
required to build a single-file pybind11 extension, and keeping the build
to plain setuptools keeps this reproducible without an extra system
dependency.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "mc_pricer_cpp",
        ["cpp/mc_pricer.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="mc_pricer_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
