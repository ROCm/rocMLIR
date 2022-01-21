#!/bin/bash
#this is to test
#
function usage() {
    echo "$0 [-s|-d] [-r|--release] RELEASE_BRANCH [-d|--directory] DIR"
    exit 1
}

function main() {
    local pwd=$(pwd) 
    local source=0
    local binary=0
    local opts
    opts=$(getopt -n "$0" -o d:r:sb --long directory:,release: -- "$@")
    if [[ $? -ne 0 ]]; then
        usage
    fi
    eval set -- "$opts"
    while : 
    do
        case "$1" in
            -s ) source=1; shift ;;
            -b ) binary=1; shift ;;
            -d | --directory ) mlir_dir="$2"; shift 2 ;;
            -r | --release )  release="$2"; shift 2 ;;
            --) shift; break ;;
            *) echo "$0: Unexpected option $1"; usage ;;
        esac
    done
    branch=${release////-}

    cd $mlir_dir
    # Create a souce package
    if [[ $source == 1 ]]; then
        echo 'Creating a source package...'
        tar --exclude=build -cvzf $pwd/mlir-source-$branch.tar.gz $(ls)
    fi

    # Create a binary package
    if [[ -d "build" && $binary == 1 ]]; then
        echo 'Creating a binary package...'
        mv build mlir-$branch
        echo "This directory contains MLIR binaries for the release branch $release." > mlir-$branch/README
        tar cvzf $pwd/mlir-binary-$branch.tar.gz ./mlir-$branch/bin ./mlir-$branch/lib ./mlir-$branch/include\
                                            ./mlir-$branch/external/llvm-project/llvm/bin\
                                            ./mlir-$branch/external/llvm-project/llvm/lib\
                                            ./mlir-$branch/external/llvm-project/llvm/include -C $mlir_dir
        mv mlir-$branch build
    fi
    cd $pwd
    echo "DONE"
}

main "$@"
