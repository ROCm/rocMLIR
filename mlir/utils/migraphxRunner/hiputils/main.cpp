#include <iostream>
#include "hip/hip_runtime.h"


int main() {
  int deviceId{0};
  hipGetDevice(&deviceId);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, deviceId);
  std::cout << "gcnArch: " << prop.gcnArch << std::endl;
  std::cout << "numCU: " << prop.multiProcessorCount << std::endl;
  std::cout << "computeMode: " << prop.computeMode << std::endl;

  return 0;
}
