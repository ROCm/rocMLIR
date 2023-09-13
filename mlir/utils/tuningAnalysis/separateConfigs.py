import os
import shutil

"""
usage: python3 separateConfigs.py

"""

def separateConfigs(operation):
    source = f"{operation}Configs"
    source_folder = os.path.dirname(source)
    output_folder = os.path.join(source_folder, operation, "separatedConfigs")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print("Remove exsiting data")
    os.makedirs(output_folder)

    with open(source, 'r') as input:
        line_num = 1
        for line in input:
            line = line.strip()

            if not line or line.startswith("#"):
                line_num+=1
                continue

            with open(f'{output_folder}/one_config_{line_num}', 'w') as output:
                output.write(line)
            
            line_num+= 1

        print("DONE!")

separateConfigs('conv')
separateConfigs('gemm')
