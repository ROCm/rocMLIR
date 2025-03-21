import os
import subprocess
from hip import hip

# Helper function to decode arch to its features
# Keep this in sync with mlir/lib/Dialect/Rock/Generator/AmdArchDb.cpp:mlir::rock::lookupArchInfo
def get_arch_features(arch: str):
    chip_name = arch.split(':')[0]
    if len(chip_name) < 5:
        return

    arch_features = None
    support_mfma = False
    support_wmma = False
    major = chip_name[:-2]
    minor = chip_name[-2:]
    if major == 'gfx9':
        if minor in ['08', '0a', '42']:
            arch_features = 'mfma|dot|atomic_add|atomic_add_f16'
        elif minor == '50':
            arch_features = 'mfma|dot|atomic_add|atomic_add_f16|atomic_add_bf16'
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
    elif major == 'gfx12':
        arch_features = 'dot|atomic_add|atomic_add_f16|atomic_add_bf16|atomic_fmax_f32|wmma'
    if arch_features and 'mfma' in arch_features:
        support_mfma = True
        pass
    elif arch_features and 'wmma' in arch_features:
        support_wmma = True
        pass
    return arch_features, support_mfma, support_wmma

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

def get_agents():
    agents = set()
    device_count = hip_check(hip.hipGetDeviceCount())
    for device in range(device_count):
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,device))
        agent = props.gcnArchName.decode('utf-8')
        agents.add(agent)

    return agents


def is_xdlops_present() -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    return any([agent.startswith("gfx9") for agent in get_agents()])
