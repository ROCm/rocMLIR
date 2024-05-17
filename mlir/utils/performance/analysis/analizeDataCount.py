import argparse
import glob
import pandas as pd

"""
usage: python3 analizeDataCount.py -o [output_file] -op [conv/gemm]

"""

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', required=True, help="output file path")
parser.add_argument('-op', '--operation', required=True, help="set operation: conv, gemm")
args = parser.parse_args()

tsv_files = pd.DataFrame()

if args.operation == "conv":
    tsv_files = glob.glob('conv/tunedData/*.debug')            
elif args.operation == 'gemm':
    tsv_files = glob.glob('gemm/tunedData/*.debug')          
else:
     raise Exception("Operation not recognized")

dfs = []

def get_max_tflops_perfconfig(group):
    max_row = group.loc[group['TFlops'].idxmax()]
    return max_row['PerfConfig']

with open(args.output, 'w') as f:
    for file in tsv_files: 
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    unique_data_types = final_df['DataType'].unique()
    top_perfconfigs_by_dataype = {}

    for data_type in unique_data_types:

        f.write(f"Data_type --> {data_type}\n")

        current_df = final_df[final_df['DataType'] == data_type]

        problem_cols = []
        if args.operation == "conv":
            problem_cols = ['N', 'C', 'K', 'Y', 'X', 'DilationH', 'DilationW', 'StrideH', 'StrideW', 'PaddingH', 'PaddingW' ]
        elif args.operation == 'gemm':
            problem_cols = ['TransA', 'TransB', 'G', 'M', 'K', 'N']
        else:
            raise Exception("Operation not recognized")

        grouped = current_df.groupby(problem_cols)
        win_counts = {}

        for name, group_df in grouped:
            max_tflops_perfconfig = get_max_tflops_perfconfig(group_df)
            if max_tflops_perfconfig not in win_counts:
                win_counts[max_tflops_perfconfig] = 0
            win_counts[max_tflops_perfconfig] += 1

        sorted_win_counts = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)   
        all_problems = set(final_df[problem_cols].apply(tuple, axis=1))
        winning_perfconfigs = set(sorted_win_counts)

        for perfconfig, count in sorted_win_counts:
            f.write(f"Perfconfig: {perfconfig}, top count: {count}\n") 
            