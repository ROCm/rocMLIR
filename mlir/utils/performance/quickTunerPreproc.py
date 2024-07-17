#!/usr/bin/env python3

"""
quickTuner preprocessor script to combine .debug output files from tuningRunner.py or tuna-script.sh

Usage: quickTunerPreprocess.py [-h] --input-dir INPUT_DIR --output OUTPUT [--op {gemm,conv}] [-d] [--file-ext FILE_EXT]

Example Usage:

python3 quickTunerPreprocess.py --input_dir /path/to/debug/files --ouput combined_data 


Note:
If using MITuna edit MITuna/tuna/rocmlir/rocmlir_worker.py, editing:


    cmd = env_str + f" python3 ./bin/tuningRunner.py -q {special_args} \
                     --config='{config_string}' --mlir-build-dir `pwd` \
                     --output=- --tflops \
                     --rocmlir_gen_flags='--device={self.gpu_id}' 2>/dev/null"


to:
    
    import uuid

    if not os.path.exists("./run"):
      os.makedirs("./run")

    unique_file_id = uuid.uuid4().hex

    file_id = os.path.join("./run", unique_file_id)
    
    cmd = env_str + f" python3 ./bin/tuningRunner.py -q {special_args} \
                     --config='{config_string}' --mlir-build-dir `pwd` \
                     --output={file_id} --tflops --debug \
                     --rocmlir_gen_flags='--device={self.gpu_id}' 2>/dev/null"

"""

import os
import sys
import argparse
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler

class qtPreprocessor(object):
    """
    class to process *.debug files into a single script
    """

    def __init__(self, pargs):
        self.input_dir = pargs.input_dir

    @staticmethod
    def __get_stats_gemm(df, ct):
        """
        static helper method to get stats for a dataframe:
        (number of files processed, number of unique gemms, group by datatype)
        """
        print(f"Files processed: {ct}")

        # num of dtypes
        dtypes= {t[0]:df for t,df in df.groupby(['DataType'])}

        print("Types found:")
        for dt in dtypes:
            print(f"\t{dt}")

        # num unique gemms in file:
        cols = ['TransA', 'TransB', 'G', 'M', 'N','K']
        unique_gemms = df[cols].drop_duplicates()
        
        num_gemms = len(unique_gemms)
        print(f"Number of unique Gemms: {num_gemms}")
        for _,row in unique_gemms.iterrows():
            tup = tuple(row)
            print(f"{tup[0]},{tup[1]},{tup[2]},{tup[3]},{tup[4]},{tup[5]}")
        

    @staticmethod
    def __get_stats_conv(df, ct):
        """
        static helper method to get stats for a dataframe:
        (number of files processed, number of unique gemms, group by datatype)
        """
        raise NotImplementedError()
        
    @staticmethod
    def process(input_dir, output_name=None, op='gemm', file_ext="debug", debug=False, normalize=True):
        """
        staticmethod process() function that compiles output files into a single dataframe and saves to tsv file
        """

        tsv_files = glob.glob(os.path.join(input_dir, f"*.{file_ext}"))
        print(os.path.join(input_dir, f"*.{file_ext}"))

        dfs = []
        ct = 0
        for file in tsv_files:
            df = pd.read_csv(file, sep='\t')
            if normalize:
                scaler = MinMaxScaler()
                df['TFlops'] = scaler.fit_transform(df[['TFlops']])
            dfs.append(df)
            ct += 1
        if not dfs:
            return None
        new_df = pd.concat(dfs, ignore_index=True)

        if output_name:
            new_df.to_csv(output_name, sep='\t')
            if debug:
                print(f"Saved to {output_name}")
                              
        if debug:            
            # here output some stats about files
            if op == 'gemm':
                qtPreprocessor.__get_stats_gemm(new_df, ct)
            elif op == 'conv':
                qtPreprocessor.__get_stats_conv(new_df, ct)

        return new_df
            
            
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='quickTunerPreprocess.py',
                                     description='Collect *.debug files from tuningRunner.py into a single file to be used in quickTunerGen.py')

    parser.add_argument('--input-dir',
                        required=True,
                        type=str,
                        help='Input directory where files are saved')

    parser.add_argument('--output',
                        required=True,
                        type=str,
                        help='File to save data to')

    parser.add_argument('--op',
                        choices=['gemm', 'conv'],
                        default='gemm',
                        help='Formats debug print info')

    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Prints debug information')

    parser.add_argument('--file-ext',
                        default='debug',
                        type=str,
                        help='File extension')

    parser.add_argument('--normalize',
                        default=True,
                        action='store_true',
                        help='Normalize on a per-file basis, necessary for quickTunerGen to work')
    
    pargs = parser.parse_args()


    qtPreprocessor.process(pargs.input_dir, pargs.output, pargs.op, pargs.file_ext, pargs.debug)
    
if __name__ == '__main__':
    main(sys.argv[1:])
