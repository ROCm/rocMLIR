#===-- Dockerfile --------------------------------------------------------===//
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===//
# Docker image used for the mlir-rocm-mi200 builder
#
# Environment variables configurable at runtime:
#    BUILDBOT_MASTER - server host to connect to
#    BUILDBOT_PORT   - server port to connect to
#===----------------------------------------------------------------------===//

# Use the rocm/mlir image as base
FROM rocm/mlir:rocm6.2-latest

RUN apt-get update; \
    apt-get install -y dumb-init;

# LTS releases often bundle obsolete pip versions that cannot access newest
# Linux binary wheels.
RUN python3 -m pip install --upgrade pip && python3 -m pip install --ignore-installed PyYAML

# Refer to mlir/lib/Bindings/Python/requirements.txt. Listed explicitly here
# and version pinned for consistency as this is a bot.
RUN python3 -m pip install numpy 'pybind11>=2.8.0' dataclasses

# Install build bot (server was at 2.8.5-dev at time of writing).
RUN pip3 install buildbot-worker

# Workaround permissions issues when writing to named volumes
# https://github.com/docker/compose/issues/3270#issuecomment-206214034
RUN mkdir -p /vol/test /vol/ccache /vol/worker ; \
    chmod -R 777 /vol

# Volume to mount secrets into the container.
VOLUME /vol/secrets
# Volume to store data for local, manual testing of the container.
VOLUME /vol/test
# Volume to store ccache.
VOLUME /vol/ccache
ENV CCACHE_DIR=/vol/ccache
# Volume for worker working directory.
VOLUME /vol/worker

# Create user account, some tests fail if run as root.
RUN useradd buildbot --create-home -u 2000
WORKDIR /vol/worker

# copy startup script
COPY run.sh /home/buildbot/
RUN chmod a+rx /home/buildbot/run.sh

USER buildbot
ENV WORKER_NAME="mi200-buildbot"

# Allow the server port of this agent to be configurable during deployment.
# This way we can connect the same image to production and integration.
# Ports:
#   9990 - production
#   9994 - integration
#   9989 - testing
ENV BUILDBOT_PORT="9989"
ENV BUILDBOT_MASTER="localhost"

CMD /home/buildbot/run.sh
