import subprocess

# Helper function to decode arch to its features
# Keep this in sync with mlir/lib/Dialect/Rock/Generator/AmdArchDb.cpp:mlir::rock::lookupArchInfo
def get_arch_features(arch: str):
    chip_name = arch.split(':')[0]
    if(len(chip_name) < 5):
        return

    arch_features = None
    support_mfma = False
    support_wmma = False
    major = chip_name[:-2]
    minor = chip_name[-2:]
    if major == 'gfx9':
        if minor in ['08', '0a', '40', '41', '42']:
            arch_features = 'mfma|dot|atomic_add'
        elif minor == '06':
            arch_features = 'dot'
        else:
            arch_features = 'none'
    elif major == 'gfx10':
        if minor in ['11', '13']:
            arch_features = 'atomic_fmax_f32'
        elif minor in ['10', '12'] or minor[0] == '3':
            arch_features = 'dot|atomic_fmax_f32'
        else:
            arch_features = 'atomic_fmax_f32'
    elif major == 'gfx11':
        arch_features = 'dot|atomic_add|atomic_fmax_f32|wmma'
    if arch_features and 'mfma' in arch_features:
        support_mfma = True
        pass
    elif arch_features and 'wmma' in arch_features:
        support_wmma = True
        pass
    return arch_features, support_mfma, support_wmma

def get_agents(rocm_path):
    p = subprocess.run([rocm_path + "/bin/rocm_agent_enumerator", "-name"],
                       check=True, stdout=subprocess.PIPE)
    agents = set(x.decode("utf-8") for x in p.stdout.split())
    if not agents:
        # TODO: Remove this workaround for a bug in rocm_agent_enumerator -name
        # Once https://github.com/RadeonOpenCompute/rocminfo/pull/59 lands
        q = subprocess.run([rocm_path + "/bin/rocm_agent_enumerator"],
                            check=True, stdout=subprocess.PIPE)
        agents = set(x.decode("utf-8") for x in q.stdout.split())
    return set(a for a in agents if a != b"gfx000")
