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

"""

To do:
- convert to base class with derived quickTuner methods
- plug in runner

"""

"""
Validator Base Class
"""
class perfConfigValidator(object):
    """
    base class for validators, implement validate() method
    """
    def __init__(self, name=None):
        self.name = name

    def collectGemmConfigs(self, gemm_config_file, comment='#'):
        """
        collect gemm config files given a gemm_config_file
        """
        gemm_list = []
        with open(gemm_config_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if comment is not None:
                line = line.split(comment)[0]

            line = line.strip()
            if line:
                gemm_list.append(line)

        return gemm_list


    def __typeQtMap(self, qt_dict):
        raise NotImplementedError()
    
    def validate(self, results):
        """
        return a dictionary of dictionaries
        """
        raise NotImplementedError()

    def getBest(self):
        """
        get the best
        """
        raise NotImplementedError()

class dataValidator(perfConfigValidator):
    """
    uses already provided data to validate the configs generated
    """
    def __init__(self, input_dir, gemm_config_file):
        super().__init__()
        self.gemm_configs = super().collectGemmConfigs(gemm_config_file)
        self.validation_data = orderByGemmType(input_dir, True)        


    def __gemmConfigToKey(self, gemm_config):
        """
        Convert gemm line to code
        """
        pattern = r'-transA (\S+) -transB (\S+) -g (\d+) -m (\d+) -n (\d+) -k (\d+)'

        match = re.search(pattern, gemm_config)

        if match:
            tup = match.groups()
            transA = True if tup[0].lower() == 'true' else False
            transB = True if tup[1].lower() == 'true' else False
            return (transA, transB) + tuple([int(x) for x in tup[2:]])
            g = match.group(3)
            m = match.group(4)
            n = match.group(5)
            k = match.group(6)
            file_name = f"g{g}_m{m}_n{n}_k{k}"
        else:
            print("Could not parse gemmConfig", file=sys.stderr)
            exit(1)
        
        return file_name


    def validate(self, results, dtype):
        all_data = []
        columns = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll']
        for gemm in self.gemm_configs:
            gemm = self.__gemmConfigToKey(gemm)
            data_subset = self.validation_data[dtype][gemm]
            results = results[list(columns)]
            merged_df = pd.merge(results, data_subset, on=['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll'], how='left')
            all_data.append(merged_df)
        return all_data
            
    

class quickTunerMethod(object):
    """
    base class for creating quick tuner methods, implement the getConfig() method.
    """
    def __init__(self, name=None):
        self.N = 40
        self.config = None
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
            

    def setN(self, N):
        """
        To set the current N count (number of configs)
        """
        self.N = N

    def saveQt(self, name=None, debug=False, suffix=".qt"):
        """
        Function to convert a type dictionary config to a .qt file
        Converts the list of quickTuning sets into a group of files
        """
        type_df = self.config
        if name is None:
            name = self.name
        if debug:
            print(filename + suffix)
            printConfigDict(type_df)
        for t in type_df:
            fname = name + "." + t + suffix
            df = type_df[t]
            if 'performance' in df.columns:
                df = df.drop(labels=['performance'], axis=1)
            df = df.to_csv(fname, index=False)

    def savePerfConfig(self, name=None,  dtype=None, prefix="v2:"):
        """
        """
        
        type_df = self.config
        if name is None:
            name = self.name

        if dtype:
            with open(name, 'w') as f:
                for _, row in type_df[dtype].iterrows():
                    tup = tuple(row)
                    s = prefix+",".join(map(str,tup))
                    f.write(s)
                    f.write("\n")
                
        else:
            for t in type_df:
                with open(f"{name}_{dtype}", 'w') as f:
                    for _, row in type_df[t].iterrows():
                        tup = tuple(row)
                        s = prefix+",".join(map(str,tup))
                        f.write(s)
                        f.write("\n")
                    
                

    def getConfig(self, input_dir):
        """
        produces a config that can be converted to .qt file using
        convertToConfig
        """
        raise NotImplementedError()


class quickTuner(object):
    """
    quickTuner class to run quick tuning methods from, requires user to instantiate quickTuner object
    then register quickTunerMethod child classes, finally run tune()
    """
    def __init__(self, pargs):
        self.methods = {}
        self.input_dir = pargs.input_dir
        self.__parseValidationArgs(pargs)

    def __parseValidationArgs(self, pargs):
        """
        parses pargs.validate string for validator
        """
        kwargs = {}
        for item in pargs.vargs:
            if '=' in item:
                k, v = item.split('=', 1)
                kwargs[k] = v
            else:
                raise ValueError(f"Argument {item} is not a valid key=value pair")

        if pargs.validate == 'data':
            # init validator
            self.validator = dataValidator(pargs.input_dir,**kwargs)
        else:
            self.validator = None               
        
    def updateInputDir(self, input_dir):
        self.input_dir = input_dir

    def addMethod(self, method: quickTunerMethod):
        """
        Adds method to method dict
        """
        self.methods[method.name] = method

    def tune(self):
        self.method_results = {}
        if not self.methods:
            print("No methods are registered, use quickTuner.addMethod(method: quickTunerMethod), to add a method", file=sys.stderr)
            exit(1)
        else:
            for k in self.methods:
                method = self.methods[k]
                df = method.getConfig(self.input_dir)
                self.method_results[k] = df

    def validate(self):
        """
        Validate on either a dataset or by running rocmlir-tuning-gen
        
        """
        if self.validator is None:
            print("validator not set", file=sys.stderr)
            return
        output_dict = {}
        for method in self.method_results:
            # df will be of the form: {type1: [data], type2: [data], ..., typeN: [data]}
            for dtype in self.method_results[method]:
                if dtype not in output_dict:
                    output_dict[dtype] = {}
                gemm_data = self.validator.validate(self.method_results[method][dtype], dtype)

                
                for df in gemm_data: # for every gemm config we get data back
                    ct = 0
                    max_values = []
                    threshold = 0.92
                    for df in gemm_data:
                        if (df['performance'].dropna() <= threshold).all():
                            #print(f"{name} does not meet threshold (>0.8): {df}")
                            ct += 1
                            #max_values.append(df[column].max())
                    output_dict[dtype][method] = ct
            
        self.output_df = pd.DataFrame(output_dict)
        print(self.output_df)
        
    
    def saveConfigs(self):
        """
        Iterate through methods and save to each file
        """
        for k in self.methods:
            method = self.methods[k]
            method.saveQt()

    def saveBest(self):
        """
        Save the best method
        """
        df = self.output_df
        
        min_values = df.min()
        best_methods = df.idxmin()

        method_counts = best_methods.value_counts()
        
        max_count = method_counts.max()
        majority_methods = method_counts[method_counts == max_count].index

        result_methods = {}
        for col in df.columns:
            candidates = df.loc[majority_methods, col]
            result_methods[col] = candidates.idxmin()
            
        # Create a list of tuples with index and corresponding method
        output = [(index, method) for index, method in result_methods.items()]

        for entry in output:
            dtype, method = entry
            self.methods[method].savePerfConfig(f"quick_tuning_{dtype}", dtype)
            
            

            
            
"""
Common methods
"""

def orderDict(type_dict: dict):
    """
    order dictionary, removing nan along the way
    """

    for k,v in type_dict.items():
        df = type_dict[k]
        #df = df.dropna(how='any')

        type_dict[k] = df.sort_values(by=['performance'], ascending=False, ignore_index=True)
        
    return type_dict

def orderGemmDict(type_gemm_dict: dict):
    """
    order type dictionary with sub dict with gemms, removing nan along the way
    """
    for k, v in type_gemm_dict.items():
        for sub_dict in v:
            df = v[sub_dict]
            df = df.dropna(how='any')

            type_gemm_dict[k][sub_dict] = df.sort_values(by=['performance'], ascending=False, ignore_index=True)

    return type_gemm_dict

def parseData(file):
    """
    reads a file then returns a dataframe containing the 
    perf config data
    """
    data = pd.read_csv(file,
                       delim_whitespace=True,
                       header=None,
                       names=['perf_config', 'performance'],
                       comment='#')
    
    data['perf_config'] = data['perf_config'].str.split(':').str[1]
                
    tile_params = data['perf_config'].str.split(',', expand=True).astype(int)
            
    tile_params.columns = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']
    
    
    tile_params = tile_params.drop(['param8','param9'], axis=1)            
            
    tile_params['performance'] = data['performance']

    tile_params.replace('N/A', np.nan, inplace=True)    

    return tile_params

def printConfigDict(df):
    for k, v in df.items():
        print(f"k:{k}\nv\n:{v}")


def readDirCluster(input_dir: str, clustered_dfs):
    """
    Given an input directory and the cluster dataframe,
    read the cluster's by datatype into a dataframe, order
    the data and take some portion of the maxes
    """
    label_dict = {}
    for label, cluster_df in clustered_dfs.items():
        # iterate through the cluster and grab all the files needed
        type_dict = {}
        for _, row in cluster_df.iterrows():
            g = row['g']
            m = row['m']
            n = row['n']
            k = row['k']
            transA = "true" if row['transA'] == 1 else 'false'
            transB = "true" if row['transB'] == 1 else 'false'

            # glob for the files
            #glob_str = f"*/*-{transA}_{transB}__g{g}_m{m}_n{n}_k{k}"
            dir_str = f"g{g}_m{m}_n{n}_k{k}"
            #glob_path = os.path.join(input_dir, glob_str)
            dir_path = os.path.join(input_dir, dir_str)        
            
            #for file in glob.glob(glob_path):
            for root, _, files in os.walk(dir_path):
                for file in files:

                    if f"-{transA}_{transB}_" in file:
                        basename = os.path.basename(file)
                        type_str = basename.split('-')[0].split('_')
                        in_type_str = type_str[0]
                        out_type_str = type_str[1]
                        if in_type_str != out_type_str:
                            continue
                    
                        tile_params = parseData(os.path.join(root,file))
                    
                        if in_type_str not in type_dict:
                            type_dict[in_type_str] = [tile_params]
                        else:
                            type_dict[in_type_str].append(tile_params)
                        
            
        for k in type_dict:
            type_dict[k] = pd.concat(type_dict[k])
        label_dict[label] = type_dict

    return label_dict


def parseDir(input_dir: str, normalize=True):

    df_dir = {}

    tsv_files = glob.glob(f"{input_dir}/*.debug")

    for file in tsv_files:
        df = pd.read_csv(file, sep='\t')
        if normalize:
            scaler = MinMaxScaler()
            df['TFlops'] = scaler.fit_transform(df[['TFlops']])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    trans_cols = ['TransA', 'TransB']

    param_cols = [ 'G', 'M', 'N','K']

    final_df = final_df.astype({entry: bool for entry in trans_cols})

    final_df = final_df.astype({entry: int for entry in param_cols})
        
    target_cols = trans_cols + param_cols

    group_df = {dtype: df for dtype, df in final_df[target_cols].groupby('DataType')}

    return group_df


def parseDir2(input_dir: str):
    """
    parse directory and make dataframe from the Gemm configs
    """

    df_dir = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            root_name = os.path.basename(root)

            parts = file.split('__')
            prefix = parts[0]
            header = parts[1].split('_')

            g, m, n, k = map(lambda x: int(x[1:]), header)

            t1_t2, transAB = prefix.split('-')

            t1, t2 = t1_t2.split('_')
            transA, transB = transAB.split('_')

            data = {}

            data['g'] = g
            data['m'] = m
            data['n'] = n
            data['k'] = k

            data['transA'] = 1 if transA.lower()  == 'true' else 0
            data['transB'] = 1 if transB.lower() == 'true' else 0

            #print(data)
            
            if t1 not in df_dir:
                df_dir[t1] = []                
            df_dir[t1].append(data)

    for k,v in df_dir.items():
        return pd.DataFrame(df_dir[k])


def orderByType(input_dir: str, normalize=False):
    df_dir = {}

    # glob the files
    tsv_files = glob.glob(f"{input_dir}/*.debug")

    dfs = []

    for file in tsv_files:
        df = pd.read_csv(file, sep='\t')
        if normalize:
            scaler = MinMaxScaler()
            df['TFlops'] = scaler.fit_transform(df[['TFlops']])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    unique_data_types = final_df['DataType'].unique()

    perf_config_cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']

    perf_configs = final_df['PerfConfig'].str.split(':').str[1].str.split(',', expand=True).astype(int)

    perf_configs.columns = perf_config_cols

    perf_configs.drop(['param8', 'param9'], axis=1, inplace=True)

    perf_configs['performance'] = final_df['TFlops']

    perf_configs['DataType'] = final_df['DataType']

    if normalize:
        scaler = MinMaxScaler()
        perf_configs['performance'] = scaler.fit_transform(perf_configs[['performance']])    
    
    result = {dtype: group.drop(['DataType'], axis=1) for dtype, group in perf_configs.groupby('DataType')}

    return result
        

def orderByType2(input_dir: str, normalize=False):
    """
    return a dictionary keyed by datatype with values of an ordered dataframe
    of the input_directory
    """
    df_dir = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            root_name = os.path.basename(root)

            parts = file.split('__')
            prefix = parts[0]
            header = parts[1].split('_')

            t1 = prefix.split('-')[0].split('_')[0]

            data = pd.read_csv(file_path,
                               delim_whitespace=True,
                               header=None,
                               names=['perf_config', 'performance'],
                               comment='#')

            data['perf_config'] = data['perf_config'].str.split(':').str[1]
            
            tile_params = data['perf_config'].str.split(',', expand=True).astype(int)
            
            tile_params.columns             = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']#= ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9']


            tile_params = tile_params.drop(['param8','param9'], axis=1)            
            
            tile_params['performance'] = data['performance']

            tile_params.replace('N/A', np.nan, inplace=True)


            if normalize:
                scaler = MinMaxScaler()
                tile_params['performance'] = scaler.fit_transform(tile_params[['performance']])
            

            if t1 not in df_dir:
                df_dir[t1] = [tile_params]
            else:
                df_dir[t1].append(tile_params)

    for k,v in df_dir.items():
        df_dir[k] = pd.concat(df_dir[k], ignore_index=True)

    return df_dir

def orderByGemmType(input_dir: str, normalize=True):

    def col2str(col):
        TransA = 't' if col[0] == 'True' else 'f'
        TransB = 't' if col[1] == 'True' else 'f'
        G = col[2]
        M = col[3]
        K = col[4]
        N = col[5]
        return f""
        
    df_dir = {}

    tsv_files = glob.glob(f"{input_dir}/*.debug")

    dfs = []

    fx = lambda x: x == 'True'

    for file in tsv_files:        
        df = pd.read_csv(file, sep='\t')
        if normalize:
            scaler = MinMaxScaler()
            df['TFlops'] = scaler.fit_transform(df[['TFlops']])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    trans_cols = ['TransA', 'TransB']

    param_cols = [ 'G', 'M', 'N','K']

    final_df = final_df.astype({entry: bool for entry in trans_cols})

    final_df = final_df.astype({entry: int for entry in param_cols})
        
    target_cols = trans_cols + param_cols

    perf_config_cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']

    #perf_configs = final_df['PerfConfig'].str.split(':').str[1].str.split(',', expand=True).astype(int)

    #perf_configs.columns = perf_config_cols

    #perf_configs.drop(['param8', 'param9'], axis=1, inplace=True)

    #final_df.drop(['PerfConfig'], axis=1, inplace=True)

    #final_df = final_df.join(perf_configs)

    #final_df = final_df.rename(columns={'TFlops': 'performance'})

    perf_configs = final_df['PerfConfig'].str.split(':').str[1].str.split(',', expand=True).astype(int)

    perf_configs.columns = perf_config_cols

    perf_configs.drop(['param8', 'param9'], axis=1, inplace=True)

    perf_configs['performance'] = final_df['TFlops']

    perf_configs = perf_configs.join(final_df[target_cols + ['DataType']])

    grouped = {dtype[0]: df.drop('DataType', axis=1) for dtype, df in perf_configs.groupby(['DataType'])}

    for k in grouped:
        # group by the gemm
        group = {cols: df.drop(target_cols, axis=1) for cols, df in grouped[k].groupby(target_cols)}
        grouped[k] = group
        
    #for k in ordered_by_gemm:
    #    print(k)
    return grouped
    

def orderByGemmType2(input_dir: str, normalize=True):
    """
    return a dictionary keyed by datatype with values of an ordered dataframe
    of the input_directory
    """
    df_dir = {}

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            root_name = os.path.basename(root)

            #print(file_path)

            parts = file.split('__')
            prefix = parts[0]
            header = parts[1].split('_')

            # only keep f32_f32, f16_f16, i8_f32
            t1 = prefix.split('-')[0].split('_')[0]

            data = pd.read_csv(file_path,
                               delim_whitespace=True,
                               header=None,
                               names=['perf_config', 'performance'],
                               comment='#')

            data['perf_config'] = data['perf_config'].str.split(':').str[1]
            
            tile_params = data['perf_config'].str.split(',', expand=True).astype(int)
            
            tile_params.columns = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']#= ['param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9']


            tile_params = tile_params.drop(['param8','param9'], axis=1)            
            
            tile_params['performance'] = data['performance']

            tile_params.replace('N/A', np.nan, inplace=True)

            if normalize:
                scaler = MinMaxScaler()
                tile_params['performance'] = scaler.fit_transform(tile_params[['performance']])

            if t1 not in df_dir:
                df_dir[t1] = {}
            df_dir[t1][root_name] = tile_params

    return df_dir

def convertToConfig(type_df, filename, suffix=".qt", debug=False):
    """
    Converts the list of quickTuning sets into a group of files
    """
    if debug:
        print(filename + suffix)
        printConfigDict(type_df)
    
    for t in type_df:
        fname = filename + "." + t + suffix
        df = type_df[t]
        if 'performance' in df.columns:
            df = df.drop(labels=['performance'], axis=1)
        df['forceUnroll'] = 1
        df = df.to_csv(fname, index=False)

        

"""
Default tuner method
"""

class defaultQuickTune(quickTunerMethod):
    """
    Default quick tune method, uses preset values for the config file
    """
    def __init__(self, name=None):
        super().__init__(name)
        self.default_f32 = pd.DataFrame({
            "M/block": [256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16],
            "N/block": [256, 64, 128, 128, 128, 64, 64, 64, 64, 64, 32, 16, 256, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 16, 128, 128, 64, 64, 32, 32, 16, 16, 32, 32, 16, 16],
            "K/block": [2, 8, 8, 4, 2, 8, 8, 8, 4, 2, 4, 4, 8, 4, 4, 4, 2, 8, 8, 8, 8, 4, 4, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 4, 8, 4, 8],
            "M/wave": [128, 128, 64, 128, 32, 64, 32, 32, 32, 128, 128, 32, 64, 64, 64, 32, 32, 32, 16, 32, 16, 32, 16, 64, 32, 16, 16, 16, 32, 16, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16],
            "N/wave": [32, 32, 16, 32, 32, 16, 32, 16, 32, 32, 16, 16, 16, 32, 16, 16, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
            "kPack": [4, 1, 4, 4, 8, 1, 4, 1, 4, 4, 4, 8, 4, 1, 4, 4, 8, 4, 4, 4, 8, 4, 8, 8, 8, 4, 4, 8, 1, 4, 4, 4, 8, 4, 8, 8, 4, 8, 4, 8],
            "forceUnroll": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })

        
        self.default_f16 = pd.DataFrame({
            "M/block": [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16],
            "N/block": [256, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 32, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 32, 16, 128, 64, 64, 32, 32, 16, 128, 32, 64, 32],
            "K/block": [8, 4, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 8, 4, 8, 8, 8, 8, 4, 4, 8, 8, 8, 8, 4, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8],
            "M/wave": [64, 64, 128, 64, 32, 32, 128, 128, 64, 64, 32, 128, 32, 32, 64, 32, 32, 32, 64, 32, 32, 32, 32, 32, 16, 32, 32, 32, 32, 16, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16],
            "N/wave": [32, 32, 32, 32, 32, 16, 32, 16, 32, 16, 32, 16, 32, 32, 16, 32, 16, 16, 32, 16, 32, 32, 32, 16, 16, 32, 32, 32, 16, 16, 32, 32, 16, 32, 16, 16, 16, 16, 16, 16],
            "kPack": [4, 8, 8, 4, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 8, 8, 8, 8, 8, 4],
            "forceUnroll": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })

        self.default_i8 = pd.DataFrame({
            "M/block": [128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16],
            "N/block": [256, 128, 128, 128, 128, 64, 64, 64, 64, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64, 64, 32, 32, 32, 32, 16, 256, 256, 128, 64, 64, 64, 64, 32, 32, 16, 64, 32, 16, 16],
            "K/block": [8, 16, 8, 8, 8, 32, 8, 8, 4, 32, 16, 8, 4, 8, 16, 8, 8, 4, 4, 16, 16, 16, 8, 8, 8, 8, 16, 4, 32, 32, 16, 8, 4, 32, 16, 16, 16, 16, 32, 16],
            "M/wave": [128, 64, 128, 64, 32, 64, 32, 32, 32, 64, 32, 64, 32, 32, 32, 32, 32, 32, 32, 32, 16, 32, 16, 32, 32, 16, 32, 32, 32, 16, 32, 16, 32, 16, 16, 16, 16, 16, 16, 16],
            "N/wave": [16, 32, 16, 16, 16, 32, 32, 16, 16, 32, 16, 16, 16, 16, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
            "kPack": [4, 8, 8, 8, 16, 4, 16, 16, 16, 4, 4, 8, 16, 8, 4, 16, 16, 16, 8, 4, 16, 4, 16, 16, 8, 16, 4, 8, 4, 4, 4, 16, 8, 4, 8, 8, 4, 16, 4, 4],
            "forceUnroll": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        })

        self.config = { 'f32': self.default_f32, 'f16': self.default_f16, 'i8': self.default_i8 }

    def getConfig(self, input_dir):
        """
        returns the already made config
        """
        return self.config

"""

Place derived quickTunerMethod classes below here:

"""
    
class topNSelection(quickTunerMethod):
    """ 
    splits data by type then splits into certain percentage evenly,
    taking the top performers from each group
    """
    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def getConfig(self, input_dir):
        type_dict = orderByType(input_dir, normalize=self.normalize)

        type_dict = orderDict(type_dict)

        config_dict = {}

        for k,v in type_dict.items():
            num_segments = self.N // 2
            seg_size = len(v) // num_segments
            selected_configs = pd.concat([v.iloc[i * seg_size:(i+1) * seg_size].head(2) for i in range(num_segments)])

            config_dict[k] = selected_configs

        self.config = config_dict
        return self.config

class topMode(quickTunerMethod):
    """
    get most common of all gemms
    """
    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def getConfig(self, input_dir):
        config_dict = {}
    
        type_gemm_dict = orderByGemmType(input_dir, normalize=self.normalize)

        type_gemm_dict = orderGemmDict(type_gemm_dict)

        for k, v in type_gemm_dict.items():
            combined = []
            for sub_key in v:
                df = v[sub_key]
                sorted_df = df.sort_values(by='performance', ascending=False)
                top_20_percent_df = sorted_df.head(int(len(df) * 0.005))
                combined.append(top_20_percent_df)

            df = pd.concat(combined)

            # now we have a list of the gemms in combined
            # remove any repetitions and order by appearance
            grouped_df = df.groupby(['M/block','N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll'], as_index=False).agg({'performance': 'count'}).rename(columns={'performance': 'count'})

            result_df = pd.merge(df, grouped_df, on=['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll'])

            final_df = result_df.loc[result_df.groupby(['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll'])['performance'].idxmax()]

            final_df = final_df.sort_values(by=['count', 'performance'], ascending=[False, False])

            config_dict[k] = final_df.head(self.N)

        self.config = config_dict
        return self.config

    
class takeNEach(quickTunerMethod):
    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def getConfig(self, input_dir):
        """
        take top performers from N dataframes
        """
        config_dict = {}
    
        type_gemm_dict = orderByGemmType(input_dir, normalize=self.normalize)

        type_gemm_dict = orderGemmDict(type_gemm_dict)

        # calculate size for amount to take

        N = self.N
    
        for k, v in type_gemm_dict.items():
            sub_dict_size = len(v)
            subset_size = N // sub_dict_size
            if subset_size == 0:
                subset_size = 1

            type_df = []
            for sub_key in v:            
                # order and take top N,
                df = v[sub_key]
                df = df.sort_values(by='performance', ascending=False)
                df = df.head(subset_size)
                type_df.append(df)

            type_df = pd.concat(type_df)
            type_df = type_df.sort_values(by='performance', ascending=False)
            type_df = type_df.head(N)

            config_dict[k] = type_df

        self.config = config_dict
        return self.config


class topConfigCluster(quickTunerMethod):
    """
    Cluster each run, take sample from total
    """

    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def getConfig(self, input_dir):
        N=self.N
        n_clusters = N//2
        type_dict = orderByType(input_dir, normalize=self.normalize)

        type_dict = orderDict(type_dict)

        result_dict = {}

        features = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll']
    
        # now we have normalized data
        for k,df in type_dict.items():
            try:
                # cluster each type
                

                features = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'performance']

            
                scaler = StandardScaler()
                features_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

                
                features_scaled['performance'] = df['performance']

                # use silhouette score for optimal clustering
                silhouette_scores = []
                upper_limit = min(len(df) // 10, 20)  # Adjust 20 or the divisor based on your heuristic
                cluster_range = range(2, upper_limit)
                for n_clusters in cluster_range:
                    mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, n_init=10, random_state=42)
                    cluster_labels = mb_kmeans.fit_predict(features_scaled[features])
                    silhouette_avg = silhouette_score(features_scaled[features], cluster_labels)
                    silhouette_scores.append((n_clusters, silhouette_avg))
                df['cluster'] = mb_kmeans.fit_predict(features_scaled[features])
            
                #kmeans = KMeans(n_clusters=n_clusters)
                #df['clusters'] = kmeans.fit_predict(features_scaled[features])
                #
                #representative_set = df.groupby('cluster').apply(lambda x: x.sample(2))
                #print(representative_set)

                # get optimal clusters
                optimal_n = max(silhouette_scores, key=lambda x: x[1])[0]
                
                # run clustering with optimal n
                mb_kmeans = MiniBatchKMeans(n_clusters=optimal_n, batch_size=100, n_init=10, random_state=42)
            
                # get proper proportion use mAtH
                proportion = int(N // optimal_n)
                representative_set = df.groupby('cluster').apply(lambda x: x.nlargest(proportion, 'performance')).reset_index(drop=True)
            
                # Sort each group by 'performance' in descending order and take the top 2 rows
                #representative_set = df.groupby('cluster').apply(lambda x: x.nlargest(2, 'performance')).reset_index(drop=True)
                #print(representative_set)
            
                #representative_set = representative_set.drop(['cluster'], axis=1)

                result_dict[k] = representative_set.drop(['cluster'], axis=1)
            except Exception as e:
                print(f"Error processing type {k}: {e}", file=sys.stderr)
                continue    
    

        self.config = result_dict
        return self.config

class topGemmCluster(quickTunerMethod):
    """
    Group each GEMM config, for each cluster take the top performer,
    this allows each cluster to have similiarities between GEMMs which 
    would hopefully contribute to a similar perf config peformance.
    """

    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def getConfig(self, input_dir):        
        N = self.N
        def calculateProportions(input_list, N=40):
            # Count the occurrences of each unique value
            count_dict = {}
            for item in input_list:
                if item in count_dict:
                    count_dict[item] += 1
                else:
                    count_dict[item] = 1

            total_elements = len(input_list)

            raw_percentages = [count_dict[key] / total_elements for key in sorted(count_dict.keys())]

            rounded_percentages = [round(p, 2) for p in raw_percentages]

            total_rounded = sum(rounded_percentages)
            difference = round(1 - total_rounded, 2)

            if difference != 0:
                max_index = rounded_percentages.index(max(rounded_percentages))
                rounded_percentages[max_index] = round(rounded_percentages[max_index] + difference, 2)

            proportions = [round(p * N) for p in rounded_percentages]

            # Adjust the proportions to ensure the sum is exactly N
            total_proportions = sum(proportions)
            difference_proportions = N - total_proportions

            if difference_proportions != 0:
                max_index = proportions.index(max(proportions))
                proportions[max_index] += difference_proportions

            return proportions

        gemm_df = parseDir(input_dir)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(gemm_df)

        # Determine the optimal number of clusters using the Elbow Method
        inertia = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features_scaled)
            inertia.append(kmeans.inertia_)


        second_derivative = np.diff(np.diff(inertia))

        # The elbow point is the point with the maximum second derivative
        optimal_k = np.argmax(second_derivative) + 2

        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(features_scaled)
        labels = kmeans.labels_

        gemm_df['cluster'] = labels

        label_proportions = calculateProportions(labels, N)

        clustered_dfs = {label: gemm_df[gemm_df['cluster'] == label] for label in gemm_df['cluster'].unique()}

        # iterate through the clustered dataframes
        label_dict = readDirCluster(input_dir, clustered_dfs)
        data_dict = {}
        for label, type_dict in label_dict.items():
            #print(f"cluster: {label}")

            for dtype in type_dict:
                df = type_dict[dtype]
                df = df.sort_values(by='performance',ascending=False)            
                df = df.head(label_proportions[label])
                if dtype not in data_dict:
                    data_dict[dtype] = df
                else:
                    data_dict[dtype] = pd.concat([data_dict[dtype], df], ignore_index=True)
                    
        self.config = data_dict
        return self.config

class analyzeDataSelect(quickTunerMethod):
    """ 
    take entire set and aggregate the repeats, averaging them out/ weighing them more heavily
    """
    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize

    def __data2df(self, data):
        def split_str(s):
            return s.split(':')[-1].split(',')
        cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll', 'param8', 'param9']
        df_dict = {}
        for k in data:
            for i,n in enumerate(split_str(k)):
                col = cols[i]
                if col not in df_dict:
                    df_dict[col] = []
                df_dict[col].append(int(n))
        return pd.DataFrame(df_dict)

    def __get_value(self, data_dict, data_type, perfconfig):
        try:
            return data_dict[data_type][perfconfig]
        except KeyError:
            return -1

    # Add averge tflops to wining perfconfigs
    def __add_average_tflops(self, counted_perf,avrg_tflops):
        for datatype, value in counted_perf.items():
            for perfconfig, perf_value in value.items():
                avg_value = self.__get_value(avrg_tflops, datatype, perfconfig)
                perf_value['tflops'] = avg_value

    # Get the perfconfig with the maximum TFLOPs
    def __get_max_tflops_perfconfig(self, group):
        max_index = group['TFlops'].idxmax()
        max_row = group.loc[max_index]
        perf_config = max_row['PerfConfig']
        group.drop(max_index, inplace=True)
        return perf_config

    def __analyzeData(self, data_path, avrg_tfops_per_datatype):
        tsv_files = pd.DataFrame()
        tsv_files = glob.glob(f"{data_path}/*.debug", recursive=True)
        dfs = []
        for file in tsv_files:
            df = pd.read_csv(file, sep='\t')
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        unique_data_types = final_df['DataType'].unique()
        # Iterate through unique data type
        results = {}
        operations = ['gemm']
        for data_type in unique_data_types:
            win_counts = {}
            for operation in operations:
                current_df = final_df[final_df['DataType'] == data_type]
                problem_cols = []
                # Determine the problem columns based on operation type
                if operation == "conv":
                    problem_cols = ['N', 'C', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW']
                elif operation == 'gemm':
                    problem_cols = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
                else:
                    raise Exception("Operation not recognized")
                grouped = current_df.groupby(problem_cols)
                # Iterate through the grouped DataFrame
                for name, group_df in grouped:
                    avg_value = -1
                    max_tflops_perfconfig = {}
                    # Checking if the perfconfig is applicable to all tuned problems
                    while avg_value == -1:
                        max_tflops_perfconfig = self.__get_max_tflops_perfconfig(group_df)
                        avg_value = self.__get_value(avrg_tfops_per_datatype, data_type, max_tflops_perfconfig)
                    if max_tflops_perfconfig not in win_counts:
                        win_counts[max_tflops_perfconfig] = {'count': 0, 'tflops': 0}
                    win_counts[max_tflops_perfconfig]['count'] += 1
                results[data_type] = win_counts
        return results

    def __averagePerformance(self, data_path):
        tsv_files = glob.glob(f"{data_path}/*.debug")
        dfs = []
        for file in tsv_files:
            df = pd.read_csv(file, sep='\t')
            dfs.append(df)
        final_df = pd.concat(dfs, ignore_index=True)
        unique_data_types = final_df['DataType'].unique()
        result = {}
        # Iterating through unique data types
        for data_type in unique_data_types:
            current_df = final_df[final_df['DataType'] == data_type]
            fgroups = current_df.groupby('PerfConfig')
            not_nan_counts = {}
            mean_tflops = {}
            problems_count = 0
            # Iterating through perconfigs in gruped DataFrame
            for perfconfig, group_df in fgroups:
                if problems_count < len(group_df):
                    problems_count = len(group_df)
                not_nan_count = pd.notna(group_df['TFlops']).sum()
                not_nan_counts[perfconfig] = not_nan_count
                mean_tflops[perfconfig] = group_df['TFlops'].mean()
            sorted_counts = sorted(not_nan_counts.items(), key=lambda x: x[1], reverse=True)
            # Filtering configurations that can be applied on all problems
            top_perfconfigs = {perfconfig: mean_tflops[perfconfig] for perfconfig, count in sorted_counts if count == problems_count}
            result[data_type] = top_perfconfigs
        return result

    def getConfig(self, input_dir):
        avrg_tfops_per_datatype = self.__averagePerformance(input_dir)
        counted_win = self.__analyzeData(input_dir, avrg_tfops_per_datatype)
        self.__add_average_tflops(counted_win,avrg_tfops_per_datatype)
        sorted_data = {}
        for datatype, configs in counted_win.items():
            # Sort the configs dictionary by 'count' and 'tflops'
            sorted_configs = dict(sorted(configs.items(), key=lambda item: (-item[1]['count'], -item[1]['tflops'])))
            sorted_data[datatype] = sorted_configs

        df_dict = {}
        for datatype, value in sorted_data.items():
            df_dict[datatype] = self.__data2df(value)
            
        self.config = df_dict  
        return df_dict

class fairSelect(quickTunerMethod):
    """ 
    take entire set and aggregate the repeats, averaging them out/ weighing them more heavily
    """
    def __init__(self, name=None, normalize=True):
        super().__init__(name)
        self.normalize = normalize


    def __get_top_90_percent(self, df):
        df_sorted = df.sort_values(by='performance', ascending=False)
        #top_90_percent_count = int(0.9 * len(df_sorted))
        return df_sorted[df_sorted['performance'] >= 0.95]

    def __combine_datasets(self, dfs):
        cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll']#, 'param8', 'param9']
        combined_df = pd.concat(dfs).sort_values(by='performance', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=cols, keep='first')
        return combined_df

    def __aggregate_datasets(self, dfs):
        feature_dict = defaultdict(list)
        count_dict = defaultdict(int)
        max_label_dict = {}
        df_dict = {} # from id to dataframe

        for df in dfs:
            df_id = id(df)
            df_dict[df_id] = df
            for _, row in df.iterrows():
                feature_vector = tuple(row[:-1])  # Assuming the last column is the label
                label = row[-1]
                feature_dict[feature_vector].append(df_id)
                count_dict[feature_vector] += 1
                if feature_vector not in max_label_dict or label > max_label_dict[feature_vector]:
                    max_label_dict[feature_vector] = label

        return feature_dict, count_dict, max_label_dict, df_dict

    def __balance_datasets(self, combined_df, original_dfs):
        cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll']#, 'param8', 'param9']
        selected_features = set()
        balanced_dataset = []

        for i in range(len(original_dfs)):
            if len(balanced_dataset) >= 40:
                break
            df = original_dfs[i]

            for _, row in df.iterrows():
                #print(row)
                feature_tuple = tuple(row[cols])

                if feature_tuple not in selected_features:
                    selected_features.add(feature_tuple)
                    balanced_dataset.append(feature_tuple)
                    break

        for _, row in combined_df.iterrows():
            if len(balanced_dataset) >= 30:
                break
            feature_tuple = tuple(row[cols])

            if feature_tuple not in selected_features:
                selected_features.add(feature_tuple)
                balanced_dataset.append(row)

        balanced_dataset_df = pd.DataFrame(balanced_dataset, columns=cols)

        return balanced_dataset_df

    def __build_final_df(self, top_dfs):
        cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'forceUnroll']#, 'param8', 'param9']
        # Aggregate common feature vectors
        feature_dict, count_dict, max_label_dict, df_dict = self.__aggregate_datasets(top_dfs)
        highest_perfs = self.__combine_datasets(top_dfs)
    
        # Sort feature vectors by their count and max label
        sorted_features = sorted(count_dict.keys(), key=lambda x: (-count_dict[x], -max_label_dict[x]))

        # Initialize final dataset and keep track of added features
        final_dataset = []
        added_features = set()
        used_dfs = set()

        for feature in sorted_features:
            if feature not in added_features:
                # Find the dataframes containing this feature
                containing_dfs = feature_dict[feature]
                if not any(df_id in used_dfs for df_id in containing_dfs):
                    # Add the feature with its maximum label
                    final_dataset.append(feature)
                    added_features.add(feature)
                    # Mark the dataframes as used
                    for df_id in containing_dfs:
                        used_dfs.add(df_id)
                    # If we have used all dataframes, break the loop
                    if len(used_dfs) == len(top_dfs):
                        break
        #print(f"feature len: {len(sorted_features)}")
        top = set([id(df) for df in top_dfs])
        used = set()
        for d in final_dataset:
            for did in feature_dict[d]:
                used.add(did)
        
        diff = top.difference(used)
        for df_id in diff:
            df = df_dict[df_id]
            for _, row in df.iterrows():
                feature = tuple(row[:-1])
                if feature not in added_features:
                    added_features.add(feature)
                    final_dataset.append(feature)
                    break

        if len(final_dataset) < self.N:
            for _, row in highest_perfs.iterrows():
                feature = tuple(row[:-1])
                if feature not in added_features:
                    added_features.add(feature)
                    final_dataset.append(feature)
                    if len(final_dataset) >= self.N:
                        break
            # add more high performers
            
            
        return pd.DataFrame(final_dataset, columns=cols)

    def getConfig(self, input_dir):
        config_dict = {}
    
        type_gemm_dict = orderByGemmType(input_dir, normalize=self.normalize)

        type_gemm_dict = orderGemmDict(type_gemm_dict)

        N = self.N
        
        for dtype, dfs in type_gemm_dict.items():
            
            top_90_percent = []
            for cfg in dfs:
                df = dfs[cfg]
                top_90_percent.append(self.__get_top_90_percent(df))

            config_dict[dtype] = self.__build_final_df(top_90_percent)
        
        self.config = config_dict
        return config_dict



def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='clusterConfigs.py',
                                     description='Bunch together runs into a parallel dir named ../{DIR_NAME}-bunch')

    parser.add_argument('-i', '--input_dir',
                        required=True,
                        type=str)

    parser.add_argument('--order',
                        action='store_true',
                        help='Order the data, removing NaN')

    parser.add_argument('--dbscan',
                        action='store_true',
                        help='Perform DBSCAN on each of the dataframes')

    parser.add_argument('--normalize',
                        action='store_true',
                        default=False,
                        help='Normalize input data')

    parser.add_argument('-v', '--validate',
                        choices=['data','tuner'],
                        type=str,
                        required=False,
                        help="Specify whether validation should be used, 'data' uses provided data library to verify, 'tuner' uses tuner library"
                        )

    parser.add_argument('vargs',
                        nargs=argparse.REMAINDER,
                        help='Additional args for validator')
                        

    pargs = parser.parse_args()

    print(pargs)

    #convertToConfig(default_data, "defaultOld", debug=True)

    tuner = quickTuner(pargs)

    tuner.addMethod(defaultQuickTune('defaultNew'))

    #tuner.addMethod(analyzeDataSelect('DjordjeMethod')) # slow asf
    
    #tuner.addMethod(topNSelection('topNSelectNew'))

    #tuner.addMethod(topMode('topModeNew'))

    #tuner.addMethod(takeNEach('takeNEachNew'))

    #tuner.addMethod(topConfigCluster('confClusterNew'))

    #tuner.addMethod(topGemmCluster('gemmClusterNew')) update parseDir

    tuner.addMethod(fairSelect('fairSelect')) 

    tuner.tune()

    tuner.validate()

    #tuner.saveBest()
    tuner.saveConfigs()
    
        
if __name__ == '__main__':
    main(sys.argv[1:])



"""
TO DO

- break into three parts
- #1: preprocessor.py, take *.debug files and generate one large input file
- #2: method/tuner, take large file and run quickTuner.py as you did before with option to save output
      to a file or in a .cpp readable output/initializer
- #3: statistics, take the output from tuner and generate statistics from both data and through perf runner, do a compile loop

eat lunch
create preprocessor.py and push
edit to accept large file, change args and change output options
create script to compile and run the code 
