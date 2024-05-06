#!/usr/bin/env perl

## To use this script, you may need to manually install Set::Scalar module
## perl -MCPAN -e 'install Set::Scalar'
##
## This script extracts unique MLIR modules from a MIGraphX trace and generate
## test cases to verify that they can be successfully compiled.
##
use 5.0.30;
use strict;
use warnings;
use utf8;
use Set::Scalar;

if (scalar @ARGV < 2) {
    print "$0 trace output-prefix [--all | test-name-file]\n";
    die
}

my $all = 0;
my $testNames = "_extractedTests_";
if (@ARGV == 3 ) {
	  if ($ARGV[2] eq "--all") {
		  $all = 1;
    }
    else {
        $testNames = $ARGV[2];
    }
}

my $funcSet = Set::Scalar->new;
if (-e $testNames) {
    # Open the file that contains all existing function names,
    # read in each line and add them into funcSet
    open(my $extracted, '<', $testNames) or die "Cannot open '$testNames' $!";
    while (my $line = <$extracted>) {
        chomp($line);
        $funcSet->insert($line);
    }
    close($extracted);
}

my $prefix = $ARGV[1];
my $count = 0;
my $toPrint = 0;
my $segment;
my $mlirCommand;

# Open the trace file for reading
open(my $trace, "<:encoding(UTF-8)", $ARGV[0]) or die $!;
while (<$trace>) {
    if (/^module\s+\{/) {
        $toPrint = 0;
    }
    # Extract the function name
    if (/^\s*func\.func\s+@(\S+?)\s*\(/) {
        my $funcname = $1;
        # Remove trailing spaces
        $funcname =~ s/\s+$//;
        # Create a test case if it doesn't exist or $all=1
        if ((not $all) and ($funcSet->has($funcname))) {
            $toPrint = 0;
        }
        else {
            # Extract the arch string
            my $arch="%arch";
            if ($_ =~ /arch\s*=\s*"([^"]*)"/) {
                $arch = $1;
            }
            open($segment, ">:encoding(UTF-8)", "${prefix}_${count}.mlir") or die $!;
            print $segment "// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -kernel-pipeline full  -arch $arch --verify-passes | rocmlir-opt\n";
            print $segment "module {\n";
            $count++;
            $toPrint = 1;
        }
        # Insert the funcname into the set
        $funcSet->insert($funcname);
    }
    if (/^\}$/) {
        if ($toPrint) {
            print $segment $_;
            close($segment);
            $toPrint = 0;
        }
    }
    else {
        if ($toPrint) {
            print $segment $_;
        }
    }
}
close($trace);

# Save the extracted function names into the file
open(my $extracted, '>', $testNames) or die "Cannot open '$testNames' $!";
foreach my $n ($funcSet->members) {
    print $extracted "$n\n";
}
close($extracted);
