#!/bin/bash

set -x

declare -a ARGS=("$@")

err() {
  echo "$*" >&2
}

# Clean up the docker images
# Note: includes a workaround for the occasional hung docker daemon
start_clean_docker() {
  systemctl status docker | grep 'Active:'
  sudo /usr/bin/pkill -f docker
  sudo /bin/systemctl restart docker
  docker system prune -a -f
  systemctl status docker | grep 'Active:'
}

# Build the docker image and push it to the repository
build_rocm_image() {
  if docker build -t rocm-mlir -f ./mlir/utils/jenkins/Dockerfile ./mlir/utils/jenkins ; then
    err Docker image build failed
    exit 1
  fi

  # Target repository to push
  local docker_repository="rocm/mlir"

  local rocm_full_version rocm_short_version git_commit_hash
  rocm_full_version=$(grep "ROCM_PATH" ./mlir/utils/jenkins/Dockerfile | sed 's/.*-\([0-9][0-9]*[.][0-9][0-9.]*\)/\1/')
  rocm_short_version="rocm${rocm_full_version%.*}"
  git_commit_hash=$(git rev-parse --short HEAD)

  # Example: rocm/mlir:rocm3.7-38337614c80
  local docker_new_img_name=${docker_repository}:${rocm_short_version}-${git_commit_hash}
  # Example: rocm/mlir:rocm3.7-latest
  local docker_new_img_name_rocm_latest=${docker_repository}:${rocm_short_version}-latest
  # Example: rocm/mlir:latest
  local docker_new_img_name_latest=${docker_repository}:latest

  # Commit the most recent container to an image with the new image name
  docker tag rocm-mlir "${docker_new_img_name_rocm_latest}"
  docker tag rocm-mlir "${docker_new_img_name_latest}"
  docker tag rocm-mlir "${docker_new_img_name}"

  if docker login -u="${USER}" -p="${PASSWORD}" ; then
    err Docker login failed
    exit 1
  fi

  docker push "${docker_new_img_name}"
  docker push "${docker_new_img_name_rocm_latest}"
  docker push "${docker_new_img_name_latest}"
  docker logout
}

main() {
  if [[ " ${ARGS[*]} " =~ " --force " ]] || \
      git diff --name-only HEAD^ HEAD | grep -q "Dockerfile"; then
    echo "Start a new build"
    start_clean_docker
    build_rocm_image
  else
    echo "No dependency changes, abort build"
    exit 0
  fi
}

main
