# Goal

Enable selective patching of a source tree based on a provided list of file names and directories.

# Prerequisites

```
python -m pip install diff-match-patch
```

# Example usage
```
BASE_DIR=$HOME/llvm-project-mlir.test/
python ./patcher.py --infile ./filelist.txt \
                    --srcdir $BASE_DIR/mlir-patch \
                    --dstdir $BASE_DIR/external/llvm-project/mlir
```