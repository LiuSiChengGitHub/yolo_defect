#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

DEPS_ROOT="${YOLO_DEFECT_AARCH64_DEPS_ROOT:-${HOME}/.local/opt/yolo-defect-aarch64}"
ORT_ROOT="${YOLO_DEFECT_AARCH64_ORT_ROOT:-${DEPS_ROOT}/onnxruntime-linux-aarch64-1.19.2}"
SYSROOT="${YOLO_DEFECT_AARCH64_SYSROOT:-${DEPS_ROOT}/ubuntu-noble-opencv-4.6.0}"
DEB_CACHE_ROOT="${YOLO_DEFECT_AARCH64_DEB_CACHE_ROOT:-${HOME}/.cache/yolo-defect}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz"

fail() {
  printf '%s\n' \
    "AArch64 dependency bootstrap failed: object=$1; expected=$2; actual=$3; action=$4" >&2
  exit 1
}

need_command() {
  command -v -- "$1" >/dev/null 2>&1 ||
    fail "command $1" "available on the x86_64 host PATH" "not found" "$2"
}

usage() {
  cat <<'EOF'
YOLO Defect AArch64 target dependency bootstrap

Usage:
  bootstrap_aarch64_deps.sh doctor
  bootstrap_aarch64_deps.sh fetch
  bootstrap_aarch64_deps.sh all

This script downloads the official ARM64 ORT SDK and extracts Ubuntu ARM64
OpenCV packages into a private target sysroot. It never apt-installs OpenCV
ARM64 packages, because doing so can remove the host's amd64 OpenCV dev chain.

Prerequisite: apt must know the arm64 foreign architecture and the Ubuntu
ports repository. See docs/paths_commands.md for the one-time root commands.

Optional environment:
  YOLO_DEFECT_AARCH64_DEPS_ROOT  Parent dependency directory below $HOME.
  YOLO_DEFECT_AARCH64_ORT_ROOT   Existing or desired official ARM64 ORT root.
  YOLO_DEFECT_AARCH64_SYSROOT    Existing or desired private target sysroot.
  YOLO_DEFECT_AARCH64_DEB_CACHE_ROOT  Parent for one isolated download directory.
EOF
}

doctor_host() {
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
    fail "host" "Linux x86_64" "$(uname -s)/$(uname -m)" \
      "run this workflow inside the documented WSL2 Ubuntu host"
  for command_name in apt-cache apt-get curl dpkg dpkg-deb file find readelf \
      sed sort tar; do
    need_command "${command_name}" "install the minimal documented host tools"
  done
  dpkg --print-foreign-architectures | grep -Fxq arm64 ||
    fail "dpkg foreign architecture" "arm64 enabled" "not enabled" \
      "run the documented dpkg --add-architecture and apt source setup"
  local candidate=""
  candidate="$(apt-cache policy libopencv-core406t64:arm64 |
    sed -n 's/^[[:space:]]*Candidate:[[:space:]]*//p')"
  [[ "${candidate}" == 4.6.0+dfsg-* ]] ||
    fail "Ubuntu ARM64 OpenCV candidate" "4.6.0 from ports.ubuntu.com" \
      "${candidate:-none}" "check the arm64 Ubuntu ports source and run apt update"
  printf '[pass] host=%s/%s; arm64 OpenCV candidate=%s\n' \
    "$(uname -s)" "$(uname -m)" "${candidate}"
}

fetch_ort() {
  if [[ ! -f "${ORT_ROOT}/VERSION_NUMBER" ]]; then
    mkdir -p -- "${ORT_ROOT}"
    local archive=""
    archive="$(mktemp --tmpdir onnxruntime-linux-aarch64-1.19.2.XXXXXX.tgz)"
    curl -fL --retry 3 -o "${archive}" "${ORT_URL}"
    tar -xzf "${archive}" -C "${ORT_ROOT}" --strip-components=1
    rm -f -- "${archive}"
  fi
}

fetch_opencv_sysroot() {
  if [[ -f "${SYSROOT}/usr/include/opencv4/opencv2/core/version.hpp" &&
        -f "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0" &&
        -f "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0" &&
        -f "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.6.0" ]]; then
    return
  fi

  mkdir -p -- "${DEB_CACHE_ROOT}" "${SYSROOT}"
  local cache_root=""
  cache_root="$(cd -- "${DEB_CACHE_ROOT}" && pwd -P)"
  local download_dir=""
  download_dir="$(mktemp -d "${cache_root}/aarch64-debs.XXXXXX")"
  local simulation=""
  simulation="$(apt-get -s -o Dir::State::status=/dev/null \
    install --no-install-recommends \
    libopencv-core406t64:arm64 \
    libopencv-imgproc406t64:arm64 \
    libopencv-imgcodecs406t64:arm64)"
  local -a runtime_packages=()
  mapfile -t runtime_packages < <(
    sed -n 's/^Inst \([^ ]*:arm64\).*/\1/p' <<<"${simulation}" | sort -u)
  ((${#runtime_packages[@]} > 0)) ||
    fail "ARM64 runtime dependency closure" "one or more target packages" \
      "empty apt simulation" "check the Ubuntu ports package indexes"

  (
    cd -- "${download_dir}"
    apt-get download "${runtime_packages[@]}"
    apt-get download \
      libopencv-core-dev:arm64 \
      libopencv-imgproc-dev:arm64 \
      libopencv-imgcodecs-dev:arm64
  )

  local package=""
  local extracted=0
  while IFS= read -r -d '' package; do
    dpkg-deb -x "${package}" "${SYSROOT}"
    extracted=$((extracted + 1))
  done < <(find "${download_dir}" -maxdepth 1 -type f -name '*_arm64.deb' \
    -print0 | sort -z)
  ((extracted > 0)) ||
    fail "ARM64 .deb extraction" "at least one downloaded package" "none" \
      "inspect ${download_dir} and retry"
  case "${download_dir}" in
    "${cache_root}"/aarch64-debs.*)
      rm -rf -- "${download_dir}"
      ;;
    *)
      fail "ARM64 download cleanup" "directory below ${cache_root}" \
        "${download_dir}" "remove the unexpected directory manually"
      ;;
  esac
  printf '[pass] private target sysroot extracted from %d ARM64 packages\n' \
    "${extracted}"
}

verify_target_deps() {
  local version=""
  [[ -f "${ORT_ROOT}/VERSION_NUMBER" ]] ||
    fail "ARM64 ORT SDK" "VERSION_NUMBER" "missing at ${ORT_ROOT}" \
      "run this script with fetch"
  version="$(tr -d '[:space:]' < "${ORT_ROOT}/VERSION_NUMBER")"
  [[ "${version}" == 1.19.2 ]] ||
    fail "ARM64 ORT version" "1.19.2" "${version}" \
      "select the pinned official SDK"

  local -a libraries=(
    "${ORT_ROOT}/lib/libonnxruntime.so.1.19.2"
    "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0"
    "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0"
    "${SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.6.0")
  local library=""
  local description=""
  for library in "${libraries[@]}"; do
    [[ -f "${library}" ]] ||
      fail "target library" "existing regular file" "${library}" \
        "run this script with fetch"
    description="$(file -L -b -- "${library}")"
    [[ "${description}" == *ELF* && "${description}" == *ARM\ aarch64* ]] ||
      fail "target library architecture" "ARM aarch64 ELF" "${description}" \
        "remove the incorrect dependency root and fetch ARM64 packages"
    readelf -h "${library}" | grep -q 'Machine:.*AArch64' ||
      fail "target ELF machine" "AArch64" "${library}" \
        "select target libraries instead of host x86_64 libraries"
  done
  [[ -f "${SYSROOT}/usr/include/opencv4/opencv2/core/version.hpp" ]] ||
    fail "target OpenCV headers" "OpenCV 4.6.0 headers" "missing" \
      "run this script with fetch"
  printf '[pass] ORT_ROOT=%s\n' "${ORT_ROOT}"
  printf '[pass] AARCH64_SYSROOT=%s\n' "${SYSROOT}"
}

action="${1:-help}"
case "${action}" in
  help|-h|--help)
    usage
    ;;
  doctor)
    doctor_host
    verify_target_deps
    ;;
  fetch)
    doctor_host
    fetch_ort
    fetch_opencv_sysroot
    verify_target_deps
    ;;
  all)
    doctor_host
    fetch_ort
    fetch_opencv_sysroot
    verify_target_deps
    ;;
  *)
    fail "action" "help, doctor, fetch, or all" "${action}" \
      "run bootstrap_aarch64_deps.sh help"
    ;;
esac
