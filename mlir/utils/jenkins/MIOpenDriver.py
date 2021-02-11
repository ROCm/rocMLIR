#!/usr/bin/python3.7

import csv
import getopt
import subprocess
import sys

# global variables.
benchmarkingResultFileName = 'results.stats.csv'
roundDigits = 2

# convolution configurations.
class ConvConfiguration:
    def generateMlirDriverCommandLine(self):
        result = ''
        # set operation.
        if self.direction == 'fwd':
            result = result + '--operation conv2d'
        elif self.direction == 'bwd':
            result = result + '--operation conv2d_bwd_data'
        elif self.direction == 'wrw':
            result = result + '--operation conv2d_bwd_weight'

        # set filter layout.
        result = result + ' --fil_layout ' + self.filterLayout

        # set input layout.
        result = result + ' --in_layout ' + self.inputLayout

        # set output layout.
        result = result + ' --out_layout ' + self.outputLayout

        # set N
        result = result + ' --batchsize ' + str(self.n)

        # set C
        result = result + ' --in_channels ' + str(self.c)

        # set Hi
        result = result + ' --in_h ' + str(self.hi)

        # set Wi
        result = result + ' --in_w ' + str(self.wi)

        # set K
        result = result + ' --out_channels ' + str(self.k)

        # set Y
        result = result + ' --fil_w ' + str(self.y)

        # set X
        result = result + ' --fil_h ' + str(self.x)

        # set dilation height
        result = result + ' --dilation_h ' + str(self.dilationH)

        # set dilation width
        result = result + ' --dilation_w ' + str(self.dilationW)

        # set stride height
        result = result + ' --conv_stride_h ' + str(self.convStrideH)

        # set stride width
        result = result + ' --conv_stride_w ' + str(self.convStrideW)

        # set padding width
        result = result + ' --padding_h ' + str(self.paddingH)

        # set padding height
        result = result + ' --padding_w ' + str(self.paddingW)

        # set data type.
        result = result + ' -t ' + self.dataType

        # set XDLOPS
        if self.xdlops == True:
            result = result + ' -x2'

        return result


    def __init__(self, argv):
        # setup default values.
        self.dataType = 'f32'
        self.xdlops = True
        self.direction = 'fwd' # fwd, bwd, wrw
        self.filterLayout = 'kcyx'
        self.inputLayout = 'nchw'
        self.outputLayout = 'nkhw'
        self.n = 128
        self.c = 1024
        self.hi = 14
        self.wi = 14
        self.k = 1024
        self.y = 1
        self.x = 1
        self.convStrideH = 1
        self.convStrideW = 1
        self.paddingH = 0
        self.paddingW = 0
        self.dilationH = 1
        self.dilationW = 1
        self.group = 1

        # determine dataType from argv[1]
        if len(argv) > 1:
            if argv[1] == 'conv':
                self.dataType = 'f32'
            elif argv[1] == 'convfp16':
                self.dataType = 'f16'
            elif argv[1] == 'convbfp16':
                self.dataType = 'bf16'

        try:
            opts, args = getopt.getopt(argv[2:], "hX:F:f:I:O:n:c:H:W:k:y:x:p:q:l:j:u:v:g:")
        except getopt.GetOptError:
            print('getopt error')
            sys.exit(-1)

        for opt, arg in opts:
            if opt == '-X':
                # -X
                self.xdlops = (int(arg) != 0)
            elif opt == '-F':
                # -F
                # 1 fwd only
                # 2 bwd only
                # 4 wrw only
                # TBD:
                # 0 fwd+bwd+wrw
                # 3 fwd+bwd
                # 5 fwd+wrw
                # 6 bwd+wrw
                if arg == 1:
                    self.direction = 'fwd'
                elif arg == 2:
                    self.direction = 'bwd'
                elif arg == 4:
                    self.direction = 'wrw'
            elif opt == '-f':
                # -f
                self.filterLayout = arg
            elif opt == '-I':
                # -I
                self.inputLayout = arg
            elif opt == '-O':
                # -O
                self.outputLayout = arg
            elif opt == "-n":
                # -n
                self.n = int(arg)
            elif opt == '-c':
                # -c
                self.c = int(arg)
            elif opt == '-H':
                # -H
                self.hi = int(arg)
            elif opt == '-W':
                # -W
                self.wi = int(arg)
            elif opt == '-k':
                # -k
                self.k = int(arg)
            elif opt == '-y':
                # -y
                self.y = int(arg)
            elif opt == '-x':
                # -x
                self.x = int(arg)
            elif opt == '-u':
                # -u
                self.convStrideH = int(arg)
            elif opt == '-v':
                # -v
                self.convStrideW = int(arg)
            elif opt == '-p':
                # -p
                self.paddingH = int(arg)
            elif opt == '-q':
                # -q
                self.paddingW = int(arg)
            elif opt == '-l':
                # -l
                self.dilationH = int(arg)
            elif opt == '-j':
                # -j
                self.dilationW = int(arg)
            elif opt == '-g':
                # -g
                self.group = int(arg)
            else:
                continue

        # Ho and Wo are computed.
        self.ho = (self.hi + self.paddingH * 2 - self.y) / self.convStrideH + 1
        self.wo = (self.wi + self.paddingW * 2 - self.x) / self.convStrideW + 1

def getNanoSeconds(fileName):
    with open(fileName, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter = ',')

        row = next(reader)
        result = row['AverageNs']
        csv_file.close()
        return result

def computeTFlops(config, nanoSeconds):
    return round((2.0 * config.n * config.c * config.k * config.ho * config.wo * config.y * config.x) / (float(nanoSeconds) * 1e-9) / 1e12, roundDigits)

config = ConvConfiguration(sys.argv)
commandLineOptions = config.generateMlirDriverCommandLine()

mlirMIOpenDriverCommand = './bin/mlir-miopen-driver -ph -c ' + commandLineOptions
profilerCommand = '/opt/rocm/bin/rocprof --hip-trace ./bin/mlir-rocm-runner --shared-libs=./lib/librocm-runtime-wrappers.so,./lib/libmlir_runner_utils.so --entry-point-result=void'

p1 = subprocess.Popen(mlirMIOpenDriverCommand.split(), stdout=subprocess.PIPE)
p2 = subprocess.Popen(profilerCommand.split(), stdin=p1.stdout, stdout=subprocess.PIPE)
p1.stdout.close() # Allow p1 to receive a SIGPIPE if p2 exits.
p2.communicate()

nanoSeconds = getNanoSeconds(benchmarkingResultFileName)
print('TFlops:', computeTFlops(config, nanoSeconds))
