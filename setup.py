###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.
# Copyright (c) 2025 Xflops - All rights reserved.
#
# For information on the license, see the LICENSE file.
# SPDX-License-Identifier: BSD-3-Clause
#
# Authors: Dhiraj Kalamkar (Intel Corp.)
#          Dragon Archer (Xflops)
###############################################################################

import os
import glob
from setuptools import setup
from setuptools import Command
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from subprocess import check_call
import pathlib
import torch

cwd = os.path.dirname(os.path.realpath(__file__))

# set debug_trace_tpp = True to enable call tracing inside extension
# export TPP_DEBUG_TRACE=0 (default) No logging (may add little overhead though)
# export TPP_DEBUG_TRACE=1 to log TPP creation
# export TPP_DEBUG_TRACE=2 to log previous and scope tracing
# export TPP_DEBUG_TRACE=3 to log previous and TPP call tracing
debug_trace_tpp = False

libxsmm_root = os.path.join(cwd, "libxsmm")
if "LIBXSMM_ROOT" in os.environ:
    libxsmm_root = os.getenv("LIBXSMM_ROOT")

xsmm_makefile = os.path.join(libxsmm_root, "Makefile")
xsmm_include = os.path.join(libxsmm_root, "include")
xsmm_lib = os.path.join(libxsmm_root, "lib")

if not os.path.exists(xsmm_makefile):
    raise IOError(
        f"{xsmm_makefile} doesn't exists! Please initialize libxsmm submodule using"
        + "    $git submodule update --init"
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class BuildMakeLib(Command):

    description = "build C/C++ libraries using Makefile"

    def initialize_options(self):
        self.build_clib = None
        self.build_temp = None

        # List of libraries to build
        self.libraries = None

        # Compilation options for all libraries
        self.define = None
        self.debug = None
        self.force = 0

    def finalize_options(self):
        self.set_undefined_options(
            "build",
            ("build_temp", "build_temp"),
            ("debug", "debug"),
            ("force", "force"),
        )
        # When building multiple third party libraries, we have to put the created lbiraries all in one place
        # (pointed to as self.build_clib) because only this path is added to the link line for extensions
        self.final_common_libs_dir = "third_party_libs"  # at the level of build_temp
        self.build_clib = self.build_temp + "/" + self.final_common_libs_dir
        self.libraries = self.distribution.libraries

    def run(self):
        pathlib.Path(self.build_clib).mkdir(parents=True, exist_ok=True)
        if not self.libraries:
            return
        self.build_libraries(self.libraries)

    def get_library_names(self):
        if not self.libraries:
            return None

        lib_names = []
        for lib_name, makefile, build_args in self.libraries:
            lib_names.append(lib_name)
        return lib_names

    def get_source_files(self):
        return []

    def build_libraries(self, libraries):
        for lib_name, makefile, build_args in libraries:
            build_dir = pathlib.Path(self.build_temp + "/" + lib_name)
            build_dir.mkdir(parents=True, exist_ok=True)
            check_call(["make", "-f", makefile] + build_args, cwd=str(build_dir))
            # NOTE: neither can use a wildcard here nor mv (since for the second library directory will already exist)
            # This copying/hard linking assumes that the libraries are putting libraries under their respective /lib subfolder
            check_call(
                ["cp", "-alf", lib_name + "/lib/.", self.final_common_libs_dir],
                cwd=str(self.build_temp),
            )
            # remove dynamic libraries to force static linking
            check_call(
                ["rm", "-f", "libxsmm.so"],
                cwd=str(self.build_clib),
            )


# AlphaFold sources
sources = glob.glob("src/csrc/*.cpp")

extra_compile_args = [
    "-fopenmp",
    "-g",
    "-DLIBXSMM_DEFAULT_CONFIG",
    "-march=native",
    "-fext-numeric-literals",
    "-O3",
    "-funroll-loops",
    "-DDYNAMIC_TILING=1",
]

if hasattr(torch, "float8_e5m2") and hasattr(torch, "float8_e4m3fn"):
    extra_compile_args.append("-DPYTORCH_SUPPORTS_FLOAT8")

if debug_trace_tpp:
    extra_compile_args.append("-DDEBUG_TRACE_TPP")

USE_CXX_ABI = int(torch._C._GLIBCXX_USE_CXX11_ABI)

print("extra_compile_args = ", extra_compile_args)

print(sources)

setup(
    name="af3_kernels",
    version="0.0.1",
    author="Dragon Archer",
    author_email="dragon-archer@outlook.com",
    description="Kernels for AlphaFold3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/libxsmm/tpp-pytorch-extension",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License (BSD-3-Clause)",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    libraries=[
        (
            "xsmm",
            xsmm_makefile,
            [
                "CC=gcc",
                "CXX=g++",
                "AVX=3",
                "-j",
                "STATIC=1",
                "SYM=1",
            ],
        ),
    ],
    ext_modules=[
        CppExtension(
            "af3_kernels._C",
            sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[xsmm_include, "{}/src/csrc".format(cwd)],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
)
