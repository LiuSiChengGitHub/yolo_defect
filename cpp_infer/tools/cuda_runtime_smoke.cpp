#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA runtime smoke failed at " << operation << ": "
              << cudaGetErrorName(status) << " ("
              << cudaGetErrorString(status) << ")\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  int device_count = 0;
  int driver_version = 0;
  int runtime_version = 0;
  require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  require_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
  require_cuda(cudaRuntimeGetVersion(&runtime_version),
               "cudaRuntimeGetVersion");
  if (device_count < 1) {
    std::cerr << "CUDA runtime smoke failed: no CUDA device is visible.\n";
    return 1;
  }

  cudaDeviceProp properties{};
  require_cuda(cudaGetDeviceProperties(&properties, 0),
               "cudaGetDeviceProperties(0)");

  constexpr std::size_t kElementCount = 1024;
  constexpr std::size_t kByteCount = kElementCount * sizeof(unsigned int);
  void* device_buffer = nullptr;
  require_cuda(cudaMalloc(&device_buffer, kByteCount), "cudaMalloc");
  require_cuda(cudaMemset(device_buffer, 0, kByteCount), "cudaMemset");
  std::vector<unsigned int> host_buffer(kElementCount, 1U);
  require_cuda(cudaMemcpy(host_buffer.data(), device_buffer, kByteCount,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy device-to-host");
  require_cuda(cudaFree(device_buffer), "cudaFree");

  for (const unsigned int value : host_buffer) {
    if (value != 0U) {
      std::cerr << "CUDA runtime smoke failed: copied memory was not zero.\n";
      return 1;
    }
  }

  std::cout << "CUDA runtime smoke PASS\n"
            << "device_count: " << device_count << "\n"
            << "device_0: " << properties.name << "\n"
            << "compute_capability: " << properties.major << '.'
            << properties.minor << "\n"
            << "driver_version: " << driver_version << "\n"
            << "runtime_version: " << runtime_version << "\n"
            << "allocated_and_copied_bytes: " << kByteCount << "\n";
  return 0;
}
