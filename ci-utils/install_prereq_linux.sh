#!/bin/bash
# Usage: $bash install_prereq_linux.sh $INSTALL_DIR
# Default $INSTALL_DIR = ./local_install
#
if [ -z "$1" ]
then
      echo "No path to the bfio-cpp source location provided"
      echo "Creating local_install directory"
      LOCAL_INSTALL_DIR="local_install"
else
     LOCAL_INSTALL_DIR=$1
fi

mkdir -p $LOCAL_INSTALL_DIR

# Set CMAKE_INSTALL_PREFIX to /usr/local when running in GitHub Actions,
# otherwise install into LOCAL_INSTALL_DIR.
if [ -n "$SYS_INSTALL" ]; then
    DEPS_INSTALL_PREFIX="/usr/local"
else
    DEPS_INSTALL_PREFIX=$LOCAL_INSTALL_DIR
fi



curl -L https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip -o v2.12.0.zip
unzip -o v2.12.0.zip
cd pybind11-2.12.0
mkdir -p build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_PREFIX" -DPYBIND11_TEST=OFF ..
make install -j4
cd ../../


curl -L https://github.com/madler/zlib/releases/download/v1.3.1/zlib131.zip -o zlib131.zip
unzip -o zlib131.zip
cd zlib-1.3.1
mkdir -p build_man
cd build_man
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_PREFIX" ..
cmake --build .
cmake --build . --target install 
cd ../../

curl -L https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.0.zip -o 3.1.0.zip
unzip -o 3.1.0.zip
cd libjpeg-turbo-3.1.0
mkdir -p build_man
cd build_man
cmake -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_PREFIX" -DENABLE_STATIC=FALSE -DCMAKE_BUILD_TYPE=Release ..
if [[ "$OSTYPE" == "darwin"* ]]; then
  sudo make install -j4
else
  make install -j4
fi
cd ../../

curl -L  https://github.com/glennrp/libpng/archive/refs/tags/v1.6.53.zip -o v1.6.53.zip
unzip -o v1.6.53.zip
cd libpng-1.6.53
mkdir -p build_man
cd build_man
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$DEPS_INSTALL_PREFIX" ..
make install -j4
cd ../../
