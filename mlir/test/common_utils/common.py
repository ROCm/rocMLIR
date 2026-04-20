from hip import hip
from amd_arch_db import GemmFeatures, lookup_arch_info


def features_to_string(features):
    val = int(features)
    if val == 0:
        return 'none'
    names = []
    for name, member in GemmFeatures.__members__.items():
        bit = int(member)
        if bit and (val & bit):
            names.append(name.lower())
    return '|'.join(names)


def get_arch_features(arch: str):
    info = lookup_arch_info(arch)
    arch_features = features_to_string(info.default_features)
    if info.has_lds_transpose_load:
        arch_features += '|lds_transpose_load'
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
    return any(
        bool(int(lookup_arch_info(agent).default_features) & int(GemmFeatures.MFMA))
        for agent in get_agents())
