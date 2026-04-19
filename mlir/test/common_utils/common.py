from hip import hip
from amd_arch_db import GemmFeatures, lookup_arch_info

FEATURE_FLAG_NAMES = [
    (1 << 0, 'mfma'),
    (1 << 1, 'wmma'),
    (1 << 2, 'dot'),
    (1 << 3, 'atomic_add'),
    (1 << 4, 'atomic_add_bf16'),
    (1 << 5, 'atomic_add_f16'),
    (1 << 6, 'atomic_fmax_f32'),
    (1 << 7, 'direct_to_lds_32b'),
    (1 << 8, 'direct_to_lds_128b'),
]


def features_to_string(features):
    val = int(features)
    if val == 0:
        return 'none'
    return '|'.join(name for bit, name in FEATURE_FLAG_NAMES if val & bit)


def get_arch_features(arch: str):
    info = lookup_arch_info(arch)
    arch_features = features_to_string(info.default_features)
    support_mfma = bool(int(info.default_features) & int(GemmFeatures.MFMA))
    support_wmma = bool(int(info.default_features) & int(GemmFeatures.WMMA))
    support_accel_fp8 = info.has_fp8_conversion_instrs or info.has_ocp_fp8_conversion_instrs
    return arch_features, support_mfma, support_wmma, support_accel_fp8


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
        hip_check(hip.hipGetDeviceProperties(props, device))
        agent = props.gcnArchName.decode('utf-8')
        agents.add(agent)

    return agents


def get_default_agent():
    """Returns the architecture of device 0, which HIP uses by default."""
    device_count = hip_check(hip.hipGetDeviceCount())
    if device_count > 0:
        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props, 0))
        return props.gcnArchName.decode('utf-8')
    return None


def is_xdlops_present() -> bool:
    """This function checks whether a GPU with xdlops support is present"""
    return any([agent.startswith("gfx9") for agent in get_agents()])
