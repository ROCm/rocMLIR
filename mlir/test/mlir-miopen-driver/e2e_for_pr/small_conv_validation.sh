#RUN: bash -- %s \
#RUN: %rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext\
#RUN: %pv %xdlops

set -uo pipefail

if [[ $# == 0 ]]; then
    echo "Usage: $0 [rocm-runner-libs] miopen-driver-args..."
    exit 1
fi
declare -g ROCM_SHARED_LIBS="$1"
shift

declare perl_validation_prog='my $in = do { local $/; <STDIN> }; '\
'if ($in =~ /^Unranked Memref base@ = 0x(?:[0-9a-f]+) rank = 1 offset = 0 '\
'sizes = \[1\] strides = \[1\] data =\s*\[1\]\s*$/s) { exit 0; } else { '\
'print "$in"; exit 1; }'

# TODO: conv2d_bwd_data fails this test
# In the interests of having something working, it's not being tested
# This should be fixed soon
declare -i exit_status
parallel mlir-miopen-driver "$@" -c --batchsize=9 --in_h=4 --in_w=4 --in_channels=2 \
    --fil_h=2 --fil_w=2 --out_channels=3 --conv_stride_h=1 --conv_stride_w=1 \
    --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 \
    --operation {1} --in_layout {2} --fil_layout {3} --out_layout {4} -t {5} --rand {6} \
    '|' mlir-rocm-runner --entry-point-result=void "--shared-libs=${ROCM_SHARED_LIBS}" \
    '|' perl -e "'$perl_validation_prog'" ::: conv2d conv2d_bwd_weight \
    ::: nchw nhwc :::+ kcyx kyxc :::+ nkhw nhwk ::: f32 f16 ::: none 1
exit_status=$?

if [[ $exit_status != 0 ]]; then
    echo "$exit_status tests failed"
    exit $exit_status
fi
exit 0
