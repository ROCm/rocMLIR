#!/bin/bash
#===-- build_run.sh ------------------------------------------------------===//
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
# This script will build and run a docker image, using volumes for data.
# Arguments:
#     <path to Dockerfile>
#     <path containing secrets>
#     optional: <command to be executed in the container>
#===----------------------------------------------------------------------===//

set -eu

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE_NAME="${1%/}"
SECRET_STORAGE="$2"
CMD=
if [ "$#" -eq 3 ];
then
    CMD="$3"
fi

cd "${DIR}/${IMAGE_NAME}"

# Mount a volume "workertest" to persit across test runs.
# Use this to keep e.g.  a git checkout or partial build across runs
if [[ $(docker volume ls | grep workertest | wc -l) == 0 ]] ; then
    docker volume create workertest
fi

# Volume to presist the build cache e.g. ccache or sccache.
# This will speed up local testing.
if [[ $(docker volume ls | grep workercache | wc -l) == 0 ]] ; then
    docker volume create workercache
fi

# Define arguments for mounting the volumes
# These differ on Windows and Linux
VOLUMES="-v ${SECRET_STORAGE}:/vol/secrets -v workertest:/vol/test -v workercache:/vol/ccache"
if [ -n "${OS+x}" ] && [[  "${OS}" == "Windows_NT" ]] ; then
    VOLUMES="-v ${SECRET_STORAGE}:c:\\volumes\\secrets -v workertest:c:\volumes\\test -v workercache:c:\sccache"
fi

# Set container arguments, if they are set in the environment:
ARGS=""
if [ -n "${BUILDBOT_PORT+x}" ] ; then
    ARGS+=" -e BUILDBOT_PORT=${BUILDBOT_PORT}"
fi
if [ -n "${BUILDBOT_MASTER+x}" ] ; then
    ARGS+=" -e BUILDBOT_MASTER=${BUILDBOT_MASTER}"
fi

# Proxy needed on lockhart hosts.
# Define it here rather than master.cfg to keep it internal.
case `hostname` in
    x[10]*) proxy='http://172.23.0.2:3128' ;;
    *)      proxy='' ;;
esac

if [ -n "$proxy" ]; then
    buildproxy="--build-arg http_proxy=${proxy} --build-arg https_proxy=${proxy}"
    runproxy="-e http_proxy=${proxy} -e https_proxy=${proxy}"
else
    buildproxy=''
    runproxy=''
fi

docker build ${buildproxy} -t "${IMAGE_NAME}:latest" .
docker run -it --net=host --device=/dev/kfd --device=/dev/dri --group-add video --name buildbot --restart always ${runproxy} ${VOLUMES} ${ARGS} "${IMAGE_NAME}:latest" ${CMD}
