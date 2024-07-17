#!/bin/bash

set -x
set -e

# Configurable variables
ROOT_DIR="${ROOT_DIR:-${HOME}/iree-rocMLIR}"

# Internal variables
SRC_DIR="${ROOT_DIR}/src"
BUILD_DIR="${ROOT_DIR}/build"
INSTALL_DIR="${ROOT_DIR}/pkgs"
CONFIG_DIR="${ROOT_DIR}/config"

ROCMLIR_SRC="${SRC_DIR}/rocMLIR"
MIGRAPHX_SRC="${SRC_DIR}/MIGraphX"
IREE_SRC="${SRC_DIR}/iree"

ROCMLIR_INSTALL="${INSTALL_DIR}/rocMLIR"
MIGRAPHX_INSTALL="${INSTALL_DIR}/MIGraphX"
IREE_INSTALL="${INSTALL_DIR}/iree"

function setup_dirs() {
  rm -rf ${ROOT_DIR}
  mkdir -pv ${SRC_DIR}
  mkdir -pv ${BUILD_DIR}
  mkdir -pv ${INSTALL_DIR}
  mkdir -pv ${CONFIG_DIR}
}

function clone_rocMLIR() {
  # Clone and checkout rocMLIR
  cd ${SRC_DIR}
  git clone https://github.com/ROCm/rocMLIR.git rocMLIR
  cd rocMLIR
  git checkout iree-rocMLIR
}

function clone_MIGraphX() {
  # Clone and patch MIGraphX
  cd ${SRC_DIR}
  git clone https://github.com/ROCm/AMDMIGraphX.git MIGraphX
  cd MIGraphX
  git checkout -b iree-rocMLIR 059819d0d5ecbb4ea67efc6f907eb6c9a98f59b7
  git submodule update --init
  git apply ${ROCMLIR_SRC}/iree-rocMLIR/migraphx.patch
}

function clone_iree() {
  # Clone and patch iree
  cd ${SRC_DIR}
  git clone https://github.com/iree-org/iree.git iree
  cd iree
  git checkout -b iree-rocMLIR 0934526e288315552650554da12d49b94845c7bf
  git submodule update --init
  git apply ${ROCMLIR_SRC}/iree-rocMLIR/iree.patch
}

function clone() {
  clone_rocMLIR
  clone_MIGraphX
  clone_iree
}

function build_rocMLIR() {
  cd ${BUILD_DIR}
  rm -rf build_rocMLIR
  mkdir build_rocMLIR
  cd build_rocMLIR
  cmake -S "${ROCMLIR_SRC}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FAT_LIBROCKCOMPILER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_CCACHE_BUILD=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DLLVM_LIT_ARGS="-v --xunit-xml-output test-results.xml" \
    -DCMAKE_INSTALL_PREFIX="${ROCMLIR_INSTALL}"
  ninja
  cmake --install .
}

function build_iree() {
  cd ${BUILD_DIR}
  rm -rf build_iree
  mkdir build_iree
  cd build_iree
  cmake -G Ninja -S "${IREE_SRC}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_INSTALL_PREFIX="${IREE_INSTALL}"
  ninja
  cmake --install .
  ln -sfv ${ROCMLIR_SRC}/iree-rocMLIR/iree-export.py ${IREE_INSTALL}/bin/iree-export
}

function build_MIGraphX() {
  # Install pre-reqs
  cd ${MIGRAPHX_SRC}
  sed -i '/ROCm\/rocMLIR/d' ./requirements.txt
  bash tools/install_prereqs.sh "${INSTALL_DIR}/migraphx-deps" .
  # Build MIGraphX
  cd ${BUILD_DIR}
  rm -rf build_MIGraphX
  mkdir build_MIGraphX
  cd build_MIGraphX
  cmake ${MIGRAPHX_SRC} -G Ninja \
    -DMIGRAPHX_ENABLE_MLIR=On \
    -DCMAKE_PREFIX_PATH="${ROCMLIR_INSTALL};${INSTALL_DIR}/migraphx-deps" \
    -DGPU_TARGETS="$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*')" \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
    -DCMAKE_LINKER_TYPE=LLD \
    -DCMAKE_INSTALL_PREFIX="${MIGRAPHX_INSTALL}"
  ninja
  cmake --install .
}

function build() {
  build_rocMLIR
  build_iree
  build_MIGraphX
}

function config() {
  cat > ${CONFIG_DIR}/setup-env.sh <<EOF
#!/bin/bash

export IREE_BIN="${IREE_INSTALL}/bin"
export IREE_CONFIG="${CONFIG_DIR}/iree-config.yaml" 

EOF
  chmod +x ${CONFIG_DIR}/setup-env.sh
  cp -v ${ROCMLIR_SRC}/iree-rocMLIR/iree-config.yaml ${CONFIG_DIR}/iree-config.yaml
}

function all() {
  setup_dirs
  clone
  build
  config
}

function clean() {
  rm -rf ${BUILD_DIR}
}

function purge() {
  rm -rf ${ROOT_DIR}
}

if declare -f "$1" > /dev/null
then
  "$@"
else
  echo "invalid function: '$1'" >&2
  exit 1
fi
