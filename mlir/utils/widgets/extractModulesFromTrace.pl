#!/usr/bin/env perl
use 5.0.30;
use strict;
use warnings;
use utf8;

if (scalar @ARGV != 2) {
    print "$0 [trace] [output prefix]\n";
    die
}
open(my $trace, "<:encoding(UTF-8)", $ARGV[0]) or die $!;
my $prefix = $ARGV[1];
my $count = 0;
my $toPrint = 0;
my $segment;
while (<$trace>) {
    if (/^module\s+\{/) {
        open($segment, ">:encoding(UTF-8)",
             "${prefix}_${count}.mlir") or die $!;
        $toPrint = 1;
    }
    if (/^\}$/) {
        print $segment $_;
        close($segment);
        $count++;
        $toPrint = 0;
    }
    if ($toPrint) {
        print $segment $_;
    }
}
close($trace);
