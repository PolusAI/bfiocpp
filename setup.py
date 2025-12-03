import os
import re
import sys
import versioneer
import platform
import subprocess
# from pathlib import Path

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


# def patch_grpc_basic_seq(build_temp: str) -> None:
#     """
#     Patch gRPC's basic_seq.h to drop the 'template' keyword from
#     Traits::template CallSeqFactory(...) which appears to upset
#     Apple Clang on macOS 15 / Xcode 16.
#     """
#     build_temp_path = Path(build_temp)
#     basic_seq_files = list(build_temp_path.rglob("basic_seq.h"))

#     if not basic_seq_files:
#         print("[patch_grpc_basic_seq] No basic_seq.h found under", build_temp_path)
#         return

#     pattern = "Traits::template CallSeqFactory("
#     replacement = "Traits::CallSeqFactory("

#     for hdr in basic_seq_files:
#         try:
#             text = hdr.read_text()
#         except OSError as e:
#             print(f"[patch_grpc_basic_seq] Failed to read {hdr}: {e}")
#             continue

#         if pattern not in text:
#             print(f"[patch_grpc_basic_seq] Pattern not found in {hdr}")
#             continue

#         new_text = text.replace(pattern, replacement)
#         try:
#             hdr.write_text(new_text)
#             print(f"[patch_grpc_basic_seq] Patched CallSeqFactory in {hdr}")
#         except OSError as e:
#             print(f"[patch_grpc_basic_seq] Failed to write {hdr}: {e}")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        print("-----------------__init__ (" + str(Extension) + ")")
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        print("-----------------__init__ self.sourcedir=" + str(self.sourcedir))


class CMakeBuild(build_ext):
    def run(self):
        print("-----------------CMakeBuild::run()...")

        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.24.0":
                raise RuntimeError("CMake >= 3.24.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print("-----------------CMakeBuild::build_extension()...")

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        if platform.system() == "Windows":
            print("-----------------Windows...")
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
        else:
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), versioneer.get_version()
        )

        if len(os.environ.get("CMAKE_ARGS", "")):
            cmake_args += os.environ.get("CMAKE_ARGS", "").split(" ")

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("--------------- cmake_args=" + str(cmake_args))
        print("--------------- build_args=" + str(build_args))

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )

        # if platform.system() == "Darwin":
        #     print("--------------- Applying gRPC basic_seq patch on macOS")
        #     patch_grpc_basic_seq(self.build_temp)

        if platform.system() == "Linux":
            rl = r"s/^#ifdef __has_builtin$/#if defined(__has_builtin)"
            rf = r"\&\& defined(__clang__)/"
            rsed = rl + rf
            path_header = "_deps/tensorstore-src/tensorstore/internal/type_traits.h"
            result = subprocess.call(["sed", "-i",
                                      rsed,
                                      path_header],
                                     cwd=self.build_temp, env=env)
            if result != 0:
                print("--------------- TensorStore patch failed!")

        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        print()  # Add an empty line for cleaner output


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="bfiocpp",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(dict(build_ext=CMakeBuild)),
    author="Sameeul Bashir Samee",
    author_email="sameeul.samee@axleinfo.com",
    url="https://github.com/PolusAI/bfiocpp",
    description="Tensorstore Based Backend for BFIO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src/python"),
    package_dir={"": "src/python"},
    ext_modules=[CMakeExtension("bfiocpp/libbfiocpp")],
    test_suite="tests",
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)
