import argparse
import glob
import pandas as pd

"""
usage: python3 analizeDataAvarge.py -o [output_file] 

"""

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True, help="output file path")
args = parser.parse_args()

conv_files = glob.glob('conv/tunedData/*.debug')
gemm_files = glob.glob('gemm/tunedData/*.debug')

tsv_files = conv_files + gemm_files

dfs = []

with open(args.output, 'w') as f:
    for file in tsv_files:
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)


    unique_data_types = final_df['DataType'].unique()

    top_perfconfigs_by_dataype = {}

    for data_type in unique_data_types:

        current_df = final_df[final_df['DataType'] == data_type]
        fgroups = current_df.groupby('PerfConfig')
    
        not_nan_counts = {}
        mean_tflops = {}
        problems_count = 0

        for perfconfig, group_df in fgroups:
        
            problems_count = len(group_df)
            not_nan_count = pd.notna(group_df['TFlops']).sum()
            not_nan_counts[perfconfig] = not_nan_count

            mean_tflops[perfconfig] = group_df['TFlops'].mean()

        sorted_counts = sorted(not_nan_counts.items(), key=lambda x: x[1], reverse=True)
        top_perfconfigs = [perfconfig for perfconfig, count in sorted_counts if count == problems_count]
        sorted_top_perfconfigs = sorted(top_perfconfigs,key=lambda x: mean_tflops[x], reverse=True )

        top_perfconfigs_by_dataype[data_type] = [
            {
                'config' : config,
                'mean_tflops': mean_tflops[config]
            }
            for config in sorted_top_perfconfigs
        ]

    for data_type, configs in top_perfconfigs_by_dataype.items():
        f.write(f"\nDataType: {data_type}\n")
        for entry in configs:
            f.write(f"PerfConfig: {entry['config']} with avarge TFlops: {entry['mean_tflops']}\n")
            