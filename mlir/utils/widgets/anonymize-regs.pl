#!/usr/bin/env perl
use 5.26.0;
use strict;
use warnings;

while (<>) {
    s/([vsa])\d+/$1?/g;
    s/([vsa])\[\d+:\d+\]/$1?/g;
    print;
}
