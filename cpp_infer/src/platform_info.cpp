#include "platform_info.h"

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <winternl.h>
#elif defined(__linux__)
#include <cerrno>
#include <cstring>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace yolo_defect_cpp {
namespace internal {
namespace {

constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;

[[noreturn]] void throw_platform_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Benchmark platform query failed: object " << object
          << "; expected " << expected << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::size_t fallback_logical_cpu_count() {
  const unsigned int count = std::thread::hardware_concurrency();
  return count == 0 ? 1 : static_cast<std::size_t>(count);
}

std::string peak_memory_scope() {
  return
      "process lifetime including config/session initialization, warmup, "
      "timed iterations, retained sample vectors, statistics, and benchmark "
      "harness; queried before JSON serialization/write";
}

#ifdef _WIN32

std::string wide_to_utf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int required = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
      static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) {
    return "unavailable";
  }
  std::string converted(static_cast<std::size_t>(required), '\0');
  if (WideCharToMultiByte(
          CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
          static_cast<int>(value.size()), converted.data(), required,
          nullptr, nullptr) != required) {
    return "unavailable";
  }
  return converted;
}

std::string windows_hostname() {
  wchar_t buffer[MAX_COMPUTERNAME_LENGTH + 1]{};
  DWORD length = MAX_COMPUTERNAME_LENGTH + 1;
  if (!GetComputerNameW(buffer, &length)) {
    return "unavailable";
  }
  return wide_to_utf8(std::wstring(buffer, length));
}

std::string windows_processor() {
  const DWORD required =
      GetEnvironmentVariableW(L"PROCESSOR_IDENTIFIER", nullptr, 0);
  if (required == 0) {
    return "unavailable";
  }
  std::vector<wchar_t> buffer(required);
  const DWORD written = GetEnvironmentVariableW(
      L"PROCESSOR_IDENTIFIER", buffer.data(), required);
  if (written == 0 || written >= required) {
    return "unavailable";
  }
  return wide_to_utf8(std::wstring(buffer.data(), written));
}

std::string windows_architecture() {
  SYSTEM_INFO information{};
  GetNativeSystemInfo(&information);
  switch (information.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
      return "x86_64";
    case PROCESSOR_ARCHITECTURE_ARM64:
      return "arm64";
    case PROCESSOR_ARCHITECTURE_INTEL:
      return "x86";
    default:
      return "unknown";
  }
}

std::string windows_version() {
  using RtlGetVersionFunction = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  const HMODULE module = GetModuleHandleW(L"ntdll.dll");
  if (module == nullptr) {
    return "unavailable";
  }
  const auto function = reinterpret_cast<RtlGetVersionFunction>(
      GetProcAddress(module, "RtlGetVersion"));
  if (function == nullptr) {
    return "unavailable";
  }
  RTL_OSVERSIONINFOW version{};
  version.dwOSVersionInfoSize = sizeof(version);
  if (function(&version) != 0) {
    return "unavailable";
  }
  return std::to_string(version.dwMajorVersion) + "." +
         std::to_string(version.dwMinorVersion) + "." +
         std::to_string(version.dwBuildNumber);
}

#elif defined(__linux__)

std::string linux_error(int error_number) {
  return std::to_string(error_number) + " (" +
         std::strerror(error_number) + ")";
}

#endif

}  // namespace

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#ifdef _WIN32
  if (gmtime_s(&utc, &now) != 0) {
    throw_platform_error(
        "timestamp_utc", "a UTC timestamp", "gmtime_s failed",
        "verify the Windows system clock");
  }
#elif defined(__linux__)
  if (gmtime_r(&now, &utc) == nullptr) {
    throw_platform_error(
        "timestamp_utc", "a UTC timestamp", "gmtime_r failed",
        "verify the Linux system clock");
  }
#else
  const std::tm* converted = std::gmtime(&now);
  if (converted == nullptr) {
    throw_platform_error(
        "timestamp_utc", "a UTC timestamp", "std::gmtime failed",
        "verify the platform system clock");
  }
  utc = *converted;
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

PlatformInfo collect_platform_info() {
  PlatformInfo information;
#ifdef _WIN32
  information.hostname = windows_hostname();
  information.processor = windows_processor();
  information.architecture = windows_architecture();
  information.os_name = "Windows";
  information.os_version = windows_version();
  const DWORD active_processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  information.logical_cpu_count =
      active_processors == 0
          ? fallback_logical_cpu_count()
          : static_cast<std::size_t>(active_processors);
#elif defined(__linux__)
  char hostname[256]{};
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    const int error_number = errno;
    throw_platform_error(
        "environment.machine.hostname", "a successful gethostname call",
        linux_error(error_number),
        "verify the WSL2/Linux hostname and process permissions");
  }
  hostname[sizeof(hostname) - 1] = '\0';
  if (hostname[0] == '\0') {
    throw_platform_error(
        "environment.machine.hostname", "a non-empty hostname", "empty",
        "configure a hostname for the WSL2/Linux environment");
  }

  struct utsname uts_information {};
  if (uname(&uts_information) != 0) {
    const int error_number = errno;
    throw_platform_error(
        "environment.os", "a successful uname call",
        linux_error(error_number),
        "verify the WSL2/Linux kernel interface");
  }

  information.hostname = hostname;
  information.processor = uts_information.machine;
  information.architecture = uts_information.machine;
  information.os_name = "Linux";
  information.os_version = uts_information.release;
  const long online_processors = sysconf(_SC_NPROCESSORS_ONLN);
  information.logical_cpu_count =
      online_processors > 0
          ? static_cast<std::size_t>(online_processors)
          : fallback_logical_cpu_count();
#else
  information.hostname = "unavailable";
  information.processor = "unavailable";
  information.architecture = "unknown";
  information.logical_cpu_count = fallback_logical_cpu_count();
  information.os_name = "unsupported";
  information.os_version = "unsupported";
#endif
  return information;
}

BenchmarkMemoryEvidence query_peak_process_memory() {
  BenchmarkMemoryEvidence evidence;
  evidence.scope = peak_memory_scope();
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (!GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
          sizeof(counters))) {
    throw_platform_error(
        "memory.peak_working_set", "a successful GetProcessMemoryInfo call",
        "Win32 error " + std::to_string(GetLastError()),
        "verify the Psapi runtime and process-query permissions");
  }
  evidence.supported = true;
  evidence.status = "supported";
  evidence.metric = "peak_working_set";
  evidence.bytes = static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#elif defined(__linux__)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    const int error_number = errno;
    throw_platform_error(
        "memory.peak_rss", "a successful getrusage(RUSAGE_SELF) call",
        linux_error(error_number),
        "verify the WSL2/Linux process resource interface");
  }
  if (usage.ru_maxrss <= 0) {
    throw_platform_error(
        "memory.peak_rss", "a positive ru_maxrss value in KiB",
        std::to_string(usage.ru_maxrss),
        "verify getrusage support on the Linux host");
  }
  const std::uint64_t peak_kib =
      static_cast<std::uint64_t>(usage.ru_maxrss);
  if (peak_kib > std::numeric_limits<std::uint64_t>::max() / 1024U) {
    throw_platform_error(
        "memory.peak_rss", "a ru_maxrss value representable in bytes",
        std::to_string(usage.ru_maxrss) + " KiB",
        "inspect the Linux resource-usage result");
  }
  evidence.supported = true;
  evidence.status = "supported";
  evidence.metric = "peak_rss";
  evidence.bytes = peak_kib * 1024U;
#else
  evidence.supported = false;
  evidence.status = "unsupported";
  evidence.metric = "peak_process_memory";
  evidence.reason =
      "Peak process memory is implemented only for Windows and Linux";
  return evidence;
#endif
  evidence.mebibytes =
      static_cast<double>(evidence.bytes) / kBytesPerMebibyte;
  return evidence;
}

}  // namespace internal
}  // namespace yolo_defect_cpp
