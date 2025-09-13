# python3 randomTools.py
def count_configs(filename):
    '''Count non # and non empty lines. Good for counting tuning problems in config files.'''
    count = 0
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
    return count


def test_after_separating():
    file_path1='./tier1-attention-configs'
    file_path2='./tier1-conv-configs'
    file_path3='./tier1-gemm-configs'

    attn_count = count_configs(file_path1)
    conv_count = count_configs(file_path2)
    gemm_count = count_configs(file_path3)

    print(f"Attn: {attn_count}, Conv: {conv_count}, Gemm: {gemm_count}")
    print(f"Total: {attn_count+conv_count+gemm_count}\n")
    aarushi_count = count_configs('../problem-config-tier-1-models')

    print(f"Aarushi's configs: {aarushi_count}")


if __name__ == '__main__':
    # python3 randomTools.py
    test_after_separating()