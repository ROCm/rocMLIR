#!/usr/bin/env python3

"""
quickTuner script to generate quick tuner perf configs. Uses single input file from quickTunerPreproc.py
as input.
Needs the input to be a combined normalized dataframe (default from quickTunerPreproc.py)

Usage: clusterConfigs.py [-h] --input-file INPUT_FILE [--method {default,topNSelect,topMode,takeNEach,fairSelect,hardcoded} [{default,topNSelect,topMode,takeNEach,fairSelect,hardcoded} ...]] [--save] [--debug] [--num NUM] [--perfconfig--format]

Example Usage:

python3 quickTunerGen.py --input-file TESTFILE.out --method fairSelect --save --debug --num 20

Will read TESTFILE.out then generate a quick tune list of length 20 for each datatype in TESTFILE.out. Will print these lists and save them to METHODNAME.DTYPE.qt.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import csv
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
import faulthandler
import re
import glob
import traceback

class quickTunerMethod(object):
    """
    base class for creating quick tuner methods, implement the getConfig() method.
    """
    def __init__(self, op, name=None, N=40):
        self.N = N
        self.config = None
        self.op = op
        if not name:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def __perfconfig_formatter(self, df, prefix="v2:"):
        """
        Add prefix to first column and remove header
        """
        df.iloc[:, 0] = prefix + df.iloc[:, 0].astype(str)
        return df
        

    def setN(self, N):
        """
        To set the current N count (number of configs)
        """
        self.N = N

    def saveQt(self, name=None, directory=None, debug=False, suffix=".qt", pf_format=False):
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
            if directory:
                fname = os.path.join(directory, fname)
            df = type_df[t]
            with open(fname, 'w') as f:
                row = df['PerfConfig']
                f.write("\n".join(row))            

    def savePerfConfig(self, name=None,  dtype=None, prefix="v2:"):
        """
        Saves perf configuration in the 'standard' format that can be read
        and accepted into other scripts or as an arg in rocmlir-gen
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
                

    def getConfig(self, combined_df):
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
        self.N = pargs.num
        self.directory = pargs.directory
        self.op = pargs.op
        self.input_file = pargs.input_file
        self.combined_df = pd.read_csv(self.input_file, sep='\t')
        self.__parseMethods(pargs)

    def __parseMethods(self, pargs):
        """
        parse each method in pargs.method
        """
        gen_methods = pargs.method
        for method in gen_methods:            
            if method == 'default':
                self.addMethod(defaultQuickTune(self.op, method, N=self.N))
            elif method == 'topNSelect':
                self.addMethod(topNSelection(self.op, method, N=self.N))
            elif method == 'topMode':
                self.addMethod(topMode(self.op, method, N=self.N))
            elif method == 'takeNEach':      
                self.addMethod(takeNEach(self.op, method, N=self.N))
            elif method == 'fairSelect':
                self.addMethod(fairSelect(self.op, method, N=self.N))
            else:
                raise ValueError(f"Unknown method: {method}")                            

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

    def addMethod(self, method: quickTunerMethod):
        """
        Adds method to method dict
        """
        self.methods[method.name] = method

    def tune(self):
        """
        tuner function that actually does the work of calling each registered
        quickTunerMethod class's getConfig method to generated perf configs
        """
        self.method_results = {}
        if not self.methods:
            print("No methods are registered, use quickTuner.addMethod(method: quickTunerMethod), to add a method", file=sys.stderr)
            exit(1)
        else:
            for k in self.methods:
                method = self.methods[k]
                df = method.getConfig(self.combined_df.copy())
                self.method_results[k] = df
    
    def saveConfigs(self, debug=False, pf_format=False):
        """
        Iterate through methods and save to each file
        """
        for k in self.methods:
            method = self.methods[k]
            method.saveQt(pf_format=pf_format, directory=self.directory)

    def printConfigs(self):
        """
        Print method's data
        """
        if not self.method_results:
            raise ValueError("Method results not generated")
        for k in self.method_results:
            for dtype in self.method_results[k]:
                df = self.method_results[k][dtype]
                print(f"dtype: {dtype}\n{df}\n")            
            
"""
Common methods
"""

def orderDict(type_dict: dict):
    """
    order dictionary, removing nan along the way
    """
    for k,v in type_dict.items():
        df = type_dict[k]

        type_dict[k] = df.sort_values(by=['performance'], ascending=False, ignore_index=True)
        
    return type_dict

def orderGemmDict(type_gemm_dict: dict):
    """
    order type dictionary with sub dict with gemms, removing nan along the way
    """
    for k, v in type_gemm_dict.items():
        for sub_dict in v:
            df = v[sub_dict]
            df = df.dropna(subset='performance', how='any')
            df = df.sort_values(by=['performance'], ascending=False, ignore_index=True)
            type_gemm_dict[k][sub_dict] = df

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
            
    tile_params.columns = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'splitK', 'forceUnroll', 'bCopyMore']
    
    tile_params['performance'] = data['performance']

    tile_params.replace('N/A', np.nan, inplace=True)    

    return tile_params

def printConfigDict(df):
    for k, v in df.items():
        print(f"k:{k}\nv\n:{v}")


def readDirCluster(input_file: str, clustered_dfs):
    """
    Given an input directory and the cluster dataframe,
    read the cluster's by datatype into a dataframe, order
    the data and take some portion of the maxes
    Change for combined_df
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

            dir_str = f"g{g}_m{m}_n{n}_k{k}"
            dir_path = os.path.join(input_file, dir_str)        
            
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


def parseDir(input_file: str, normalize=True):

    final_df = input_file

    trans_cols = ['TransA', 'TransB']

    param_cols = [ 'G', 'M', 'N','K']

    final_df = final_df.astype({entry: bool for entry in trans_cols})

    final_df = final_df.astype({entry: int for entry in param_cols})
        
    target_cols = trans_cols + param_cols

    group_df = {dtype: df for dtype, df in final_df[target_cols].groupby('DataType')}

    return group_df

def orderByType(combined_df: str, normalize=False):

    final_df = combined_df
    unique_data_types = final_df['DataType'].unique()

    final_df['performance'] = final_df['NormalizedTFlops']

    if normalize:
        scaler = MinMaxScaler()
        final_df['performance'] = scaler.fit_transform(final_df[['performance']])    
    
    result = {dtype: group.drop(['DataType'], axis=1) for dtype, group in final_df.groupby('DataType')}

    return result

def orderByGemmType(combined_df: str, normalize=True):

    final_df = combined_df
    
    trans_cols = ['TransA', 'TransB']

    param_cols = [ 'G', 'M', 'N','K']

    final_df = final_df.astype({entry: bool for entry in trans_cols})

    final_df = final_df.astype({entry: int for entry in param_cols})
        
    target_cols = trans_cols + param_cols

    final_df['performance'] = final_df['NormalizedTFlops']

    grouped = {dtype[0]: df.drop('DataType', axis=1) for dtype, df in final_df.groupby(['DataType'])}

    for k in grouped:
        group = {cols: df.drop(target_cols, axis=1) for cols, df in grouped[k].groupby(target_cols)}
        grouped[k] = group
        
    return grouped

def orderByConvType(combined_df: str, normalize=True):

    final_df = combined_df

    cols = ['N', 'C', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW']
    
    final_df = final_df.astype({entry: int for entry in cols})

    final_df['performance'] = final_df['NormalizedTFlops']

    grouped = {dtype[0]: df.drop('DataType', axis=1) for dtype, df in final_df.groupby(['DataType'])}

    for k in grouped:
        group = {cols: df.drop(target_cols, axis=1) for cols, df in grouped[k].groupby(cols)}
        grouped[k] = group
        
    return grouped

    

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
        df = df.to_csv(fname, index=False)

"""
Place child quickTunerMethod classes below here:
"""
    
class topNSelection(quickTunerMethod):
    """ 
    splits data by type then splits into certain percentage evenly,
    taking the top performers from each group
    """
    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.normalize = normalize

    def getConfig(self, combined_df):
        type_dict = orderByType(combined_df, normalize=self.normalize) # update this to not use columns

        type_dict = orderDict(type_dict) # change this to not rely on columns

        config_dict = {}

        for k,v in type_dict.items():
            num_segments = self.N // 2
            seg_size = len(v) // num_segments
            selected_configs = pd.concat([v.iloc[i * seg_size:(i+1) * seg_size].head(2) for i in range(num_segments)])
            config_dict[k] = selected_configs[['PerfConfig', 'performance']]

        self.config = config_dict
        return self.config

class topMode(quickTunerMethod):
    """
    Count occurrences of each perf config, take top most common
    perf configs
    """
    
    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.op = op
        self.normalize = normalize

    def getConfig(self, input_file):
        config_dict = {}

        if self.op == 'gemm':
            type_dict = orderByGemmType(input_file, normalize=self.normalize)
        elif self.op == 'conv':
            type_dict = orderByConvType(input_file, normalize=self.normalize)

        type_dict = orderGemmDict(type_dict)

        for k, v in type_dict.items():
            combined = []
            for sub_key in v:
                df = v[sub_key]
                sorted_df = df.sort_values(by='performance', ascending=False)
                top_20_percent_df = sorted_df.head(int(len(df) * 0.005))
                combined.append(top_20_percent_df)

            df = pd.concat(combined)

            # now we have a list of the gemms in combined
            # remove any repetitions and order by appearance
            grouped_df = df.groupby(['PerfConfig'], as_index=False).agg({'performance': 'count'}).rename(columns={'performance': 'count'})

            result_df = pd.merge(df, grouped_df, on=['PerfConfig'])

            final_df = result_df.loc[result_df.groupby(['PerfConfig'])['performance'].idxmax()]

            final_df = final_df.sort_values(by=['count', 'performance'], ascending=[False, False])

            config_dict[k] = final_df.head(self.N)

        self.config = config_dict
        return self.config

    
class takeNEach(quickTunerMethod):
    """
    take top performers from N dataframes
    """
    
    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.op = op
        self.normalize = normalize

    def getConfig(self, combined_df):
        config_dict = {}


        if self.op == 'gemm':
            type_dict = orderByGemmType(combined_df, normalize=self.normalize)
        elif self.op == 'conv':
            type_dict = orderByConvType(combined_df, normalize=self.normalize)

        type_dict = orderByGemmType(combined_df, normalize=self.normalize)
        

        type_dict = orderGemmDict(type_dict)

        # calculate size for amount to take

        N = self.N
    
        for k, v in type_dict.items():
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


class topConfigCluster(quickTunerMethod): #disabled
    """
    Cluster each run, take sample from total,
    can be improved via new distance metric
    for Kmeans clustering, alternatively
    try with DBSCAN again
    """
    
    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.normalize = normalize

    def getConfig(self, combined_df):
        N=self.N
        n_clusters = N//2
        type_dict = orderByType(combined_df, normalize=self.normalize)

        type_dict = orderDict(type_dict)

        result_dict = {}

        features = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'splitK', 'forceUnroll', 'bCopyMore']
    
        # now we have normalized data
        for k,df in type_dict.items():
            try:
                features = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'splitK', 'forceUnroll', 'bCopyMore', 'performance']

            
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
            
                # get optimal clusters
                optimal_n = max(silhouette_scores, key=lambda x: x[1])[0]
                
                # run clustering with optimal n
                mb_kmeans = MiniBatchKMeans(n_clusters=optimal_n, batch_size=100, n_init=10, random_state=42)
            
                # get proper proportion use mAtH
                proportion = int(N // optimal_n)
                representative_set = df.groupby('cluster').apply(lambda x: x.nlargest(proportion, 'performance')).reset_index(drop=True)
            
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

    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.normalize = normalize

    def getConfig(self, combined_df):        
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

        gemm_df = parseDir(input_file)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(gemm_df)

        # determine the optimal number of clusters using the elbow method
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
        label_dict = readDirCluster(input_file, clustered_dfs)
        data_dict = {}
        for label, type_dict in label_dict.items():

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

class defaultQuickTune(quickTunerMethod):
    """ 
    take entire set and aggregate the repeats, averaging them out/ weighing them more heavily
    """
    def __init__(self, op, name=None, N=40, normalize=True):
        super().__init__(op, name, N)
        self.op = op
        self.normalize = normalize
        self.N

    def __get_value(self, data_dict, data_type, perfconfig):
        try:
            return data_dict[data_type][perfconfig]
        except KeyError:
            return -1

    # add averge tflops to wining perf configs
    def __add_average_tflops(self, counted_perf,avrg_tflops):
        for datatype, value in counted_perf.items():
            for perfconfig, perf_value in value.items():
                avg_value = self.__get_value(avrg_tflops, datatype, perfconfig)
                perf_value['tflops'] = avg_value

    # get the perf config with the maximum tflops
    def __get_max_tflops_perfconfig(self, group):
        max_index = group['NormalizedTFlops'].idxmax()
        max_row = group.loc[max_index]
        perf_config = max_row['PerfConfig']
        group.drop(max_index, inplace=True)
        return perf_config

    def __analyzeData(self, combined_df, avrg_tfops_per_datatype):
        tsv_files = pd.DataFrame()
        final_df = combined_df
        unique_data_types = final_df['DataType'].unique()
        # iterate through unique data type
        results = {}
        operations = [self.op]
        for data_type in unique_data_types:
            win_counts = {}
            for operation in operations:
                current_df = final_df[final_df['DataType'] == data_type]
                problem_cols = []
                # determine the problem columns based on operation type
                if operation == "conv":
                    problem_cols = ['N', 'C', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW']
                elif operation == 'gemm':
                    problem_cols = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
                else:
                    raise Exception("Operation not recognized")
                grouped = current_df.groupby(problem_cols)
                # iterate through the grouped df
                for name, group_df in grouped:
                    avg_value = -1
                    max_tflops_perfconfig = {}
                    # checking if the perf config applies to all tuned perfs
                    while avg_value == -1:
                        max_tflops_perfconfig = self.__get_max_tflops_perfconfig(group_df)
                        avg_value = self.__get_value(avrg_tfops_per_datatype, data_type, max_tflops_perfconfig)
                    if max_tflops_perfconfig not in win_counts:
                        win_counts[max_tflops_perfconfig] = {'count': 0, 'tflops': 0}
                    win_counts[max_tflops_perfconfig]['count'] += 1
                results[data_type] = win_counts
        return results

    def __averagePerformance(self, combined_df):
        final_df = combined_df
        unique_data_types = final_df['DataType'].unique()
        result = {}
        # iterating through unique data types
        for data_type in unique_data_types:
            current_df = final_df[final_df['DataType'] == data_type]
            fgroups = current_df.groupby('PerfConfig')
            not_nan_counts = {}
            mean_tflops = {}
            problems_count = 0
            # iterating through perf configs in grouped dfs
            for perfconfig, group_df in fgroups:
                if problems_count < len(group_df):
                    problems_count = len(group_df)
                not_nan_count = pd.notna(group_df['NormalizedTFlops']).sum()
                not_nan_counts[perfconfig] = not_nan_count
                mean_tflops[perfconfig] = group_df['NormalizedTFlops'].mean()
            sorted_counts = sorted(not_nan_counts.items(), key=lambda x: x[1], reverse=True)
            top_perfconfigs = {perfconfig: mean_tflops[perfconfig] for perfconfig, count in sorted_counts if count == problems_count}
            result[data_type] = top_perfconfigs
        return result

    def getConfig(self, combined_df):
        avrg_tfops_per_datatype = self.__averagePerformance(combined_df)
        counted_win = self.__analyzeData(combined_df, avrg_tfops_per_datatype)
        self.__add_average_tflops(counted_win,avrg_tfops_per_datatype)
        sorted_data = {}
        for datatype, configs in counted_win.items():
            # sort the configs dictionary by 'count' and 'tflops'
            sorted_configs = dict(sorted(configs.items(), key=lambda item: (-item[1]['count'], -item[1]['tflops'])))
            sorted_data[datatype] = sorted_configs

        df_dict = {}

        for datatype, value in sorted_data.items():
            df_dict[datatype] = pd.DataFrame(value.keys(), columns=['PerfConfig'])
            
        self.config = df_dict  
        return df_dict

class fairSelect(quickTunerMethod):
    """ 
    take entire set and aggregate the repeats, averaging them out/ weighing them more heavily.
    Breakdown of steps:
    1) for type and each gemm
    2) get the top 90% of each list, found by using min-max scalar
    3) sort the features by count (occurrences) and performance
    4) iterate over sorted feature and for each feature:
           check if it has been added to the final dataset
           if not find the dataframes that contain this feature
           if none have been used add the feature to the final
           dataset, mark feature as added, mark df as used.
           if all dataframe have been used, break
    5) if any dataframes have not been represented yet, add top
       performeres from each dataframe until all are represented
    6) fill any remaining performers until required size is met
       or
       cut down spacce (df.head(N)) 
    """
    
    def __init__(self, op, name=None, N=40, normalize=True, threshold=0.95):
        super().__init__(op, name, N)
        self.op = op
        self.normalize = normalize
        self.threshold = threshold # top 95 percent for efficiency

    def __get_top_90_percent(self, df):
        df_sorted = df.sort_values(by='performance', ascending=False)
        return df_sorted[df_sorted['performance'] >= self.threshold]

    def __combine_datasets(self, dfs):
        combined_df = pd.concat(dfs).sort_values(by='performance', ascending=False)
        combined_df = combined_df.drop_duplicates(subset='PerfConfig', keep='first')
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
                #feature_vector = tuple(row[:-1])  # get feature (all but performance)
                feature_vector = row['PerfConfig']
                label = row['performance']
                feature_dict[feature_vector].append(df_id)
                count_dict[feature_vector] += 1
                if feature_vector not in max_label_dict or label > max_label_dict[feature_vector]:
                    max_label_dict[feature_vector] = label

        return feature_dict, count_dict, max_label_dict, df_dict

    def __build_final_df(self, top_dfs):
        #cols = ['M/block', 'N/block', 'K/block', 'M/wave', 'N/wave', 'kPack', 'splitK', 'forceUnroll', 'bCopyMore']
        # aggregate common feature vectors
        feature_dict, count_dict, max_label_dict, df_dict = self.__aggregate_datasets(top_dfs)

        highest_perfs = self.__combine_datasets(top_dfs)

        # sort features by count and max label
        sorted_features = sorted(count_dict.keys(), key=lambda x: (-count_dict[x], -max_label_dict[x]))

        # int final dataset and keep track of added features
        final_dataset = []
        added_features = set()
        used_dfs = set()
        

        for feature in sorted_features:
            if feature not in added_features:
                # find the dataframes containing this feature
                containing_dfs = feature_dict[feature]
                if not any(df_id in used_dfs for df_id in containing_dfs):
                    # add the feature with its maximum label
                    final_dataset.append(feature)
                    added_features.add(feature)
                    # mark the dataframes as used
                    for df_id in containing_dfs:
                        used_dfs.add(df_id)
                    # used all labels
                    if len(used_dfs) == len(top_dfs):
                        break
        top = set([id(df) for df in top_dfs])
        used = set()
        for d in final_dataset:
            for did in feature_dict[d]:
                used.add(did)
        
        diff = top.difference(used)
        for df_id in diff:
            df = df_dict[df_id]
            for _, row in df.iterrows():
                feature = row['PerfConfig']
                if feature not in added_features:
                    added_features.add(feature)
                    final_dataset.append(feature)
                    break

        if len(final_dataset) < self.N:
            for _, row in highest_perfs.iterrows():
                feature = row['PerfConfig']
                if feature not in added_features:
                    added_features.add(feature)
                    final_dataset.append(feature)
                    if len(final_dataset) >= self.N:
                        break
        
        return pd.DataFrame(final_dataset).head(self.N) # though this should really not be set

    def getConfig(self, combined_df):
        config_dict = {}

        if self.op == 'gemm':
            type_dict = orderByGemmType(combined_df, normalize=self.normalize)
        elif self.op == 'conv':
            type_dict = orderByConvType(combined_df, normalize=self.normalize)

        for dtype, dfs in type_dict.items():            
            top_90_percent = []
            for cfg in dfs:
                df = dfs[cfg]
                top_90_percent.append(self.__get_top_90_percent(df))
            df = self.__build_final_df(top_90_percent)
            df.columns = ['PerfConfig']
            config_dict[dtype] = df
        
        self.config = config_dict
        return config_dict



def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog='clusterConfigs.py',
                                     description='Bunch together runs into a parallel dir named ../{DIR_NAME}-bunch')

    parser.add_argument('--input-file',
                        required=True,
                        type=str)

    parser.add_argument('--method',
                        nargs='+',
                        choices=["default","topNSelect","topMode","takeNEach","fairSelect"],
                        default=["default","fairSelect"],
                        help='Select perfConfig gen selection method')

    parser.add_argument('--op', '--operation',
                        type=str,
                        choices=["gemm", "conv"],
                        default="gemm",
                        help='Operation (gemm or conv)')

    parser.add_argument('--save',
                        action='store_true',
                        default=False,
                        help='Save configs to name.dtype.qt')

    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help='Print debug info, print config files to stdout')

    parser.add_argument('--num', '-n',
                        type=int,
                        default=40,
                        help='Number of perf configs to include')

    parser.add_argument('--perfconfig-format',
                        action='store_true',
                        default=False,
                        help='Save file in correct csv perfconfig format')

    parser.add_argument('--directory',
                        type=str,
                        help='Directory to store results to')

    pargs = parser.parse_args()

    tuner = quickTuner(pargs)

    tuner.tune()

    if pargs.save:
        tuner.saveConfigs(pf_format=pargs.perfconfig_format)

    if pargs.debug:
        tuner.printConfigs()
            
if __name__ == '__main__':
    main(sys.argv[1:])

