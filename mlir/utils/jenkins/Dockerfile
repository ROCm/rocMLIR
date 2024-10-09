FROM ubuntu:22.04
MAINTAINER Zhuoran Yin <zhuoran.yin@amd.com>

ARG ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/6.2/
ARG ROCM_BUILD_NAME=jammy
ARG ROCM_BUILD_NUM=main
ARG ROCM_PATH=/opt/rocm-6.2

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get clean all

# --------------------- Section 1: ported from llvm-premerge-checks -----------------
# Keep this section up-to-date with the upstream
# https://github.com/google/llvm-premerge-checks/blob/main/containers/buildbot-linux/Dockerfile
RUN echo 'install build dependencies'; \
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        locales openssh-client gnupg ca-certificates apt-transport-https \
        zip wget curl git \
        gdb build-essential  \
        ninja-build \
        libelf-dev libffi-dev gcc-multilib libmpfr-dev libpfm4-dev \
        python3 python3-psutil python3-pip python3-setuptools \
        lsb-release software-properties-common \
        swig python3-dev libedit-dev libncurses5-dev libxml2-dev libxslt-dev liblzma-dev golang rsync jq \
        # for libc++ tests that use the timezone database of the chrono header
        tzdata \
        # for llvm installation script
        sudo \
        # build scripts
        nodejs ccache \
        # shell users
        less vim

# Install cmake 3.23+ from source.
RUN wget --no-verbose -O /cmake.sh https://github.com/Kitware/CMake/releases/download/v3.23.3/cmake-3.23.3-linux-x86_64.sh; \
    chmod +x /cmake.sh; \
    mkdir -p /etc/cmake; \
    /cmake.sh --prefix=/etc/cmake --skip-license; \
    ln -s /etc/cmake/bin/cmake /usr/bin/cmake; \
    ln -s /etc/cmake/bin/ctest /usr/bin/ctest; \
    cmake --version; \
    rm /cmake.sh

# LLVM must be installed after prerequisite packages.
ENV LLVM_VERSION=17
RUN echo 'install llvm ${LLVM_VERSION}' && \
    wget --no-verbose https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} && \
    apt-get update && \
    apt-get install -y clang-${LLVM_VERSION} clang-format-${LLVM_VERSION} clang-tidy-${LLVM_VERSION} lld-${LLVM_VERSION} && \
    ln -s /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang && \
    ln -s /usr/bin/clang++-${LLVM_VERSION} /usr/bin/clang++ && \
    ln -s /usr/bin/clang-tidy-${LLVM_VERSION} /usr/bin/clang-tidy && \
    ln -s /usr/bin/clang-tidy-diff-${LLVM_VERSION}.py /usr/bin/clang-tidy-diff && \
    ln -s /usr/bin/clang-format-${LLVM_VERSION} /usr/bin/clang-format && \
    ln -s /usr/bin/git-clang-format-${LLVM_VERSION} /usr/bin/git-clang-format && \
    ln -s /usr/bin/clang-format-diff-${LLVM_VERSION} /usr/bin/clang-format-diff && \
    ln -s /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld && \
    ln -s /usr/bin/lldb-${LLVM_VERSION} /usr/bin/lldb && \
    ln -s /usr/bin/ld.lld-${LLVM_VERSION} /usr/bin/ld.lld && \
    ln -s /usr/bin/llvm-profdata-${LLVM_VERSION} /usr/bin/llvm-profdata && \
    ln -s /usr/bin/llvm-cov-${LLVM_VERSION} /usr/bin/llvm-cov && \
    ln -s /usr/bin/llvm-symbolizer-${LLVM_VERSION} /usr/bin/llvm-symbolizer && \
    ln -s /usr/bin/llvm-cxxfilt-${LLVM_VERSION} /usr/bin/llvm-cxxfilt && \
    clang --version

RUN echo 'configure locale' && \
    sed --in-place '/en_US.UTF-8/s/^#//' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install python dependencies for the scripts. ADD will check contents of a file for changes.
# TODO: that should be done during the build as it will pull this repo anyway and will have latest version.
ADD "https://raw.githubusercontent.com/google/llvm-premerge-checks/master/scripts/requirements.txt" requirements.txt
RUN pip3 install -r requirements.txt

# --------------------- Section 2: Rock dialect setups -----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
  apt-utils \
  gpg \
  libnuma-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
           python3-venv \
    && apt upgrade -y \
    && python3 -m pip install --upgrade pip \
    && pip3 install pandas numpy scipy jinja2 tomli \
    && pip3 install -r requirements.txt
    # ^ pip install pandas... is needed to output performance tests and regenerate test suites
    # Has a copy of the requirements.txt install bit for the new python version

# Need "render" group because some CI hosts have /dev/kfd under it.
RUN groupadd -g 109 render

# Add ROCm build distribution
RUN wget --no-check-certificate -qO - http://repo.radeon.com/rocm/rocm.gpg.key 2>/dev/null | apt-key add -
RUN echo "deb [arch=amd64] $ROCM_DEB_REPO $ROCM_BUILD_NAME $ROCM_BUILD_NUM" > /etc/apt/sources.list.d/rocm.list
RUN echo 'Package: *' > /etc/apt/preferences.d/repo-radeon-pin-600 && \
    echo 'Pin: release o=repo.radeon.com' >> /etc/apt/preferences.d/repo-radeon-pin-600 && \
    echo 'Pin-Priority: 600' >> /etc/apt/preferences.d/repo-radeon-pin-600

# Note instead of installing the latest, we should always manually bump cmake version when necessary.
# This make sure that we don't accidentally use newer cmake features incompatible with our client.
# llvm-premerge-checks installs 3.23.3, see above.
# As of 2022-08-17, the rocm/rock images has /opt/conda/bin/cmake at 3.22.1.
# Note that the mlir minimum is 3.20.0.
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  rocm-dev \
  rocm-llvm-dev \
  rocm-cmake \
  rocminfo \
  rocprofiler-dev \
  rocblas \
  rocblas-dev \
  hipblas-dev \
  hipblaslt-dev \
  miopen-hip \
  miopen-hip-dev \
  libelf1 \
  pkg-config \
  kmod \
  file \
  netcat-openbsd \
  libsqlite3-dev \
  parallel \
  libaio1 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# mysql setup for tuna
# Latest "generic" version.  (Must have libaio1 installed already.)
RUN wget https://dev.mysql.com/get/Downloads/MySQL-8.0/mysql-8.0.34-linux-glibc2.28-x86_64.tar.gz && \
    tar xf mysql-8.0.34-linux-glibc2.28-x86_64.tar.gz && \
    mv mysql-8.0.34-linux-glibc2.28-x86_64 /usr/mysql

# python setup for tuna
# --ignore-installed because of problems upgrading PyYAML.  See also -U.
ADD "https://raw.githubusercontent.com/ROCm/MITuna/pf-tuna-rocmlir-3/requirements.txt" tuna-requirements.txt
RUN python3 -m venv /tuna-venv && . /tuna-venv/bin/activate && \
    python3 -m pip install -r tuna-requirements.txt --ignore-installed && \
    python3 -m pip install scipy pandas

# Workaround for 6.1.1 rocprof not supporting navi3x:  add it to list.
RUN sed --in-place 's/gfx94x")/gfx94x","gfx11xx")/' /opt/rocm/bin/rocprof
