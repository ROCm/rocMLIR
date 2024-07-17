#!/usr/bin/env python3

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys

sys.path.append('../..')

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import faulthandler
import re
import glob

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
    def process(input_dir, output_name=None, op='gemm', file_ext="debug", debug=False):
        """
        staticmethod process() function that compiles output files into a single dataframe and saves to tsv file
        """

        tsv_files = glob.glob(os.path.join(input_dir, f"*.{file_ext}"))
        print(os.path.join(input_dir, f"*.{file_ext}"))

        dfs = []
        ct = 0
        for file in tsv_files:
            df = pd.read_csv(file, sep='\t')
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
    
    pargs = parser.parse_args()


    qtPreprocessor.process(pargs.input_dir, pargs.output, pargs.op, pargs.file_ext, pargs.debug)
    
if __name__ == '__main__':
    main(sys.argv[1:])
