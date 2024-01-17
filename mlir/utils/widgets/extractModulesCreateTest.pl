#!/usr/bin/env perl

## To use this script, you may need to manually install Set::Scalar module
## perl -MCPAN -e 'install Set::Scalar'

use 5.0.30;
use strict;
use warnings;
use utf8;
use Set::Scalar;

if (scalar @ARGV < 2) {
    print "$0 trace output-prefix [--all]\n";
    die
}

my $all = 0;
if ((@ARGV == 3 ) && ($ARGV[2] eq "--all")) {
    $all = 1;
}

open(my $trace, "<:encoding(UTF-8)", $ARGV[0]) or die $!;
my $prefix = $ARGV[1];
my $count = 0;
my $toPrint = 0;
my $writeToFile = 0;
my $segment;
my $funcSet = Set::Scalar->new;
my $mlirCommand;
while (<$trace>) {
    if (/^module\s+\{/) {
            #open($segment, ">:encoding(UTF-8)",
            # "${prefix}_${count}.mlir") or die $!;
        $toPrint = 0;
    }
    # Extract the function name
    if (/^\s*func\.func\s+@(\S+?)\s*\(/) {
        my $funcname = $1;
        # Remove trailing spaces
        $funcname =~ s/\s+$//;
        $writeToFile = 1;
        if ((not $all) and ($funcSet->has($funcname))) {
            $toPrint = 0;
        }
        else {
            open($segment, ">:encoding(UTF-8)",
             "${prefix}_${count}.mlir") or die $!;
            print $segment "// RUN: rocmlir-gen --clone-harness -arch %arch -fut ${funcname} %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline migraphx,highlevel -targets %arch | rocmlir-gen -ph -verifier clone -fut ${funcname}_wrapper - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// ALLOW_RETRIES: 2
// CLONE: [1 1 1]\n";
            print $segment "module {\n";
            $count++;
            $toPrint = 1;
        }
        $funcSet->insert($funcname);
    }
    if (/^\}$/) {
        if ($toPrint) {
            print $segment $_;
        }
        close($segment);
        $toPrint = 0;
    }
    if ($toPrint) {
        print $segment $_;
    }
}
close($trace);
