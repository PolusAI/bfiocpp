This is the new backend for `bfio`, using `Tensorstore` and other high-throughput IO library

## Build Requirements

`bfiocpp` uses `Tensorstore` for reading and writing OME Tiff and OME Zarr files. So `Tensorstore` build requirements are needed to be satisfied for `bfiocpp` also. 
For Linux, these are the requirements:
- `GCC` 10 or later
- `Clang` 8 or later
- `Python` 3.8 or later
- `CMake` 3.24 or later
- `Perl`, for building *libaom* from source (default). Must be in `PATH`. Not required if `-DTENSORSTORE_USE_SYSTEM_LIBAOM=ON` is specified.
- `NASM`, for building *libjpeg-turbo*, *libaom*, and *dav1d* from source (default). Must be in `PATH`.Not required if `-DTENSORSTORE_USE_SYSTEM_{JPEG,LIBAOM,DAV1D}=ON` is specified.
- `GNU Patch` or equivalent. Must be in `PATH`.


Since `Tensorstore` requires `MACOSX_DEPLOYMENT_TARGET` to be `10.14` or higher (to support sized/aligned operator `new`/`delete`.), `bfiocpp` needs the same.


## Building and Installing

Here is an example of building and installing `bfiocpp` in a Python virtual environment.
```
python -m venv build_venv
source build_venv/bin/activate
git clone https://github.com/PolusAI/bfiocpp.git 
cd bfiocpp
mkdir build_deps
cd build_deps
../ci-utils/install_prereq_linux.sh
cd ..
export BFIOCPP_DEP_DIR=./build_deps/local_install
python setup.py install -vv
```