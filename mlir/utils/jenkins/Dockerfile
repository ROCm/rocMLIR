FROM ubuntu:20.04
MAINTAINER Zhuoran Yin <zhuoran.yin@amd.com>

ARG ROCM_DEB_REPO=http://repo.radeon.com/rocm/apt/5.5/
ARG ROCM_BUILD_NAME=focal
ARG ROCM_BUILD_NUM=main
ARG ROCM_PATH=/opt/rocm-5.5

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get clean all

# --------------------- Section 1: ported from llvm-premerge-checks -----------------
# Keep this section up-to-date with the upstream
# https://github.com/google/llvm-premerge-checks/blob/master/containers/base-debian/Dockerfile
RUN echo 'install build dependencies'; \
    apt-get update ;\
    apt-get install -y --no-install-recommends \
        locales openssh-client gnupg ca-certificates  \
        zip wget git \
        gdb build-essential  \
        ninja-build \
        ccache \
        python3 python3-psutil \
        python3-pip python3-setuptools \
        swig python3-dev libedit-dev libncurses5-dev libxml2-dev liblzma-dev golang rsync jq;

RUN apt-get update ;\
    apt-get upgrade -y ;\
    apt-get install -y \
        clang-10 lld-10 clang-tidy-10 clang-format-10 \
        ;\
    apt-get clean

RUN echo 'configure locale'; \
    sed --in-place '/en_US.UTF-8/s/^#//' /etc/locale.gen ;\
    locale-gen ;\
    echo 'make python 3 default'; \
    rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python; \
    pip3 install wheel

# Configure locale
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install python dependencies for the scripts. ADD will check contentents of a file for changes changed.
# TODO: that should be done during the build as it will pull this repo anyway and will have latest version.
ADD "https://raw.githubusercontent.com/google/llvm-premerge-checks/master/scripts/requirements.txt" requirements.txt
RUN pip3 install -r requirements.txt

RUN ln -s /usr/bin/clang-10 /usr/bin/clang;\
    ln -s /usr/bin/clang++-10 /usr/bin/clang++;\
    ln -s /usr/bin/clang-tidy-10 /usr/bin/clang-tidy;\
    ln -s /usr/bin/clang-tidy-diff-10.py /usr/bin/clang-tidy-diff;\
    ln -s /usr/bin/clang-format-10 /usr/bin/clang-format;\
    ln -s /usr/bin/git-clang-format-10 /usr/bin/git-clang-format;\
    ln -s /usr/bin/clang-format-diff-10 /usr/bin/clang-format-diff;\
    ln -s /usr/bin/lld-10 /usr/bin/lld;\
    ln -s /usr/bin/llvm-symbolizer-10 /usr/bin/llvm-symbolizer

# --------------------- Section 2: Rock dialect setups -----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
  apt-utils \
  apt-transport-https \
  curl \
  gpg \
  libnuma-dev \
  software-properties-common

RUN apt-get update && apt-get install -y --no-install-recommends \
           python3.8 python3.8-dev python3.8-venv python3-psutil \
           python3-pip python3-setuptools \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 \
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

# Install cmake distribution
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | \
    tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | \
    tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Note instead of installing the latest, we should always manually bump cmake version when necessary.
# This make sure that we don't accidentally use newer cmake features incompatible with our client.
# As of 2022-08-17, the rocm/rock images has /opt/conda/bin/cmake at 3.22.1, so we'll match that.
# Note that the mlir minimum is 3.18.
RUN apt-get update && rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
  apt-get install -y --no-install-recommends \
  rocm-dev \
  rocm-cmake \
  rocminfo \
  rocprofiler-dev \
  rocblas \
  rocblas-dev \
  miopen-hip \
  libelf1 \
  pkg-config \
  sudo \
  vim \
  kmod \
  file \
  netcat-openbsd \
  cmake-data=3.22.1-0kitware1ubuntu20.04.1 \
  cmake=3.22.1-0kitware1ubuntu20.04.1 \
  kitware-archive-keyring \
  libsqlite3-dev \
  parallel && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# --------------------- Section 3: MIOpen dependencies ---------------
WORKDIR /MIOpenDepsWork
RUN wget https://raw.githubusercontent.com/ROCmSoftwarePlatform/MIOpen/develop/requirements.txt \
    && wget https://raw.githubusercontent.com/ROCmSoftwarePlatform/MIOpen/develop/install_deps.cmake \
    && sed -i -e '/rocMLIR/d' requirements.txt \
    && sed -i -e '/composable_kernel/d' requirements.txt \
    && cmake -P install_deps.cmake --prefix /usr/local
RUN echo $(git ls-remote https://github.com/ROCmSoftwarePlatform/MIOpen HEAD) > miopen-deps-commit-hash
WORKDIR /
