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
  docker stop $(docker ps -a -q)
  docker rm -f $(docker ps -a -q)
  sudo /bin/systemctl restart docker
  docker system prune -a -f
  systemctl status docker | grep 'Active:'
}

# Build the docker image and push it to the repository
build_rocm_image() {
  local image_name_suffix=$1
  local docker_file="Dockerfile"
  if [ ! -z "$image_name_suffix" ]; then
    docker_file="Dockerfile.$image_name_suffix"
  fi
  echo "Start a new build for $docker_file"
  if ! docker build -t rocm-mlir -f ./mlir/utils/jenkins/$docker_file ./mlir/utils/jenkins ; then
    err Docker image build failed
    exit 1
  fi

  # Target repository to push
  local docker_repository="rocm/mlir"
  local image_name_wo_tag="$docker_repository"
  if [ ! -z "$image_name_suffix" ]; then
    image_name_wo_tag="$docker_repository-$image_name_suffix"
  fi

  local rocm_full_version rocm_short_version git_commit_hash rocm_major_version rocm_minor_version
  rocm_full_version=$(grep "ROCM_PATH" ./mlir/utils/jenkins/$docker_file | sed 's/.*-\([0-9][0-9]*[.][0-9][0-9.]*\)/\1/')
  rocm_major_version=$(echo ${rocm_full_version} | cut -d. -f1)
  rocm_minor_version=$(echo ${rocm_full_version} | cut -d. -f2)
  rocm_short_version="rocm${rocm_major_version}.${rocm_minor_version}"
  git_commit_hash=$(git rev-parse --short HEAD)

  # Example: rocm/mlir:rocm3.7-38337614c80
  local docker_new_img_name=${image_name_wo_tag}:${rocm_short_version}-${git_commit_hash}
  # Example: rocm/mlir:rocm3.7-latest
  local docker_new_img_name_rocm_latest=${image_name_wo_tag}:${rocm_short_version}-latest
  # Example: rocm/mlir:latest
  local docker_new_img_name_latest=${image_name_wo_tag}:latest

  # Commit the most recent container to an image with the new image name
  docker tag rocm-mlir "${docker_new_img_name_rocm_latest}"
  docker tag rocm-mlir "${docker_new_img_name_latest}"
  docker tag rocm-mlir "${docker_new_img_name}"

  if ! docker login -u="${DOCKER_USER}" -p="${DOCKER_PASSWORD}" ; then
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
      git diff --name-only HEAD^ HEAD | grep -iq "Dockerfile"; then
    start_clean_docker
    build_rocm_image ""
    build_rocm_image "migraphx-ci"
  else
    echo "No dependency changes, abort build"
    exit 0
  fi
}

main
