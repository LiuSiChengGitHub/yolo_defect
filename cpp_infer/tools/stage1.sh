#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CPP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${CPP_DIR}/.." && pwd -P)"
CALLER_DIR="$(pwd -P)"

CONFIG="${CPP_DIR}/configs/default_config.txt"
DEMO_IMAGE="${REPO_ROOT}/data/images/val/crazing_241.jpg"
MANIFEST="${CPP_DIR}/tests/fixtures/consistency_manifest.json"
CONSISTENCY_TOOL="${CPP_DIR}/tools/compare_consistency.py"
DETECTION_VALIDATOR="${CPP_DIR}/tests/assert_detection_json.py"
BENCHMARK_VALIDATOR="${CPP_DIR}/tests/assert_benchmark_json.py"
BATCH_VALIDATOR="${CPP_DIR}/tools/validate_batch_summary.py"
BATCH_COMPARISON_TOOL="${CPP_DIR}/tools/compare_batch_runs.py"
BATCH_MANIFEST="${CPP_DIR}/tests/fixtures/s2_03_consistency_manifest.txt"
BATCH_PERFORMANCE_INPUT="${REPO_ROOT}/data/images/val"
DETECT_ROOT="${CPP_DIR}/results/manual"

BUILD_REQUEST="${YOLO_DEFECT_BUILD_DIR:-/tmp/yolo_defect_stage1_linux_release}"
RUN_REQUEST="${YOLO_DEFECT_RUN_DIR:-}"
ORT_REQUEST="${ONNXRUNTIME_ROOT:-}"
PYTHON_REQUEST="${YOLO_DEFECT_PYTHON:-python3}"
GTEST_REQUEST="${YOLO_DEFECT_GTEST_SOURCE:-}"

BUILD_DIR=""
ORT_ROOT=""
PYTHON_EXE=""
GTEST_SOURCE=""
OPENCV_VERSION=""
OPENCV_CMAKE_DIR=""
CLI=""
IMAGE_PROBE=""
RUN_DIR=""

fail() {
  printf '%s\n' \
    "Stage-1 Linux failed: object=$1; expected=$2; actual=$3; action=$4" >&2
  exit 1
}

stage() {
  printf '\n[Stage-1 Linux] %s\n' "$1"
}

run() {
  local name="$1"
  shift
  printf '[run] %s\n' "${name}"
  if ! "$@"; then
    fail "${name}" "exit code 0" "nonzero exit" \
      "read the first diagnostic above, correct it, and rerun this action"
  fi
}

usage() {
  cat <<'EOF'
YOLO Defect Linux x86_64 workflow

Usage:
  stage1.sh help
  stage1.sh doctor
  stage1.sh build
  stage1.sh clean-build
  stage1.sh test
  stage1.sh detect <image> [output-dir] [--config <path>] [--overwrite]
  stage1.sh batch <input-dir-or-manifest> [output-dir] [--config <path>]
                  [--workers <1..64>] [--queue-capacity <1..4096>]
                  [--output-images] [--overwrite]
  stage1.sh batch-compare [--config <path>]
  stage1.sh demo
  stage1.sh consistency
  stage1.sh benchmark [--warmup <n>] [--repeat <n>]
  stage1.sh all

Actions:
  doctor       Read-only Linux/toolchain/SDK/dependency preflight.
  build        Configure when needed, then build current sources.
  clean-build  Recreate the guarded /tmp Ninja Release build.
  test         Build and run the complete CTest gate.
  detect       Write JSON and PNG for one arbitrary image.
  batch        Run one directory/path-list manifest with bounded workers.
  batch-compare Run workers=1 and workers=4, queue=8, JSON-only on all
               data/images/val images with the selected RuntimeConfig
               (default FP32); require identical detections and describe
               throughput/peak-RSS deltas without a speedup gate.
  demo         Validate the fixed crazing_241 JSON/PNG result.
  consistency  Run the frozen Python ORT/C++ ORT comparison.
  benchmark    Run consistency, then benchmark (default 10/100).
  all          Clean build -> CTest -> demo -> consistency -> benchmark.

Environment:
  ONNXRUNTIME_ROOT          Official Linux x64 ORT 1.19.2 SDK root (required).
  YOLO_DEFECT_PYTHON        Python with cv2/numpy/onnxruntime; default python3.
  YOLO_DEFECT_GTEST_SOURCE  GoogleTest source; defaults to /usr/src/googletest.
  YOLO_DEFECT_BUILD_DIR     Must be /tmp/.../yolo_defect_stage1_*.
  YOLO_DEFECT_RUN_DIR       Optional fresh durable run root.

Invoke with bash when the checkout does not preserve executable bits:
  bash cpp_infer/tools/stage1.sh doctor
EOF
}

need_command() {
  command -v -- "$1" >/dev/null 2>&1 ||
    fail "command $1" "available on PATH" "not found" "$2"
}

need_file() {
  [[ -f "$1" ]] || fail "$2" "an existing regular file" "$1" "$3"
}

absolute_file() {
  local path="$1"
  [[ "${path}" == /* ]] || path="${CALLER_DIR}/${path}"
  [[ -f "${path}" ]] ||
    fail "$2" "an existing regular file" "${path}" "$3"
  realpath -e -- "${path}"
}

absolute_output() {
  local path="$1"
  [[ "${path}" == /* ]] || path="${CALLER_DIR}/${path}"
  realpath -m -- "${path}"
}

resolve_build_dir() {
  local lexical=""
  lexical="$(realpath -m -s -- "${BUILD_REQUEST}")"
  BUILD_DIR="$(realpath -m -- "${BUILD_REQUEST}")"
  local leaf="${BUILD_DIR##*/}"
  [[ "${BUILD_DIR}" == /tmp/* &&
     "${leaf}" =~ ^yolo_defect_stage1_[A-Za-z0-9._-]+$ ]] ||
    fail "build clean boundary" "a yolo_defect_stage1_* directory below /tmp" \
      "${BUILD_DIR}" "use the default or another dedicated /tmp path"
  [[ "${lexical}" == "${BUILD_DIR}" ]] ||
    fail "build clean boundary" "a path without symbolic-link components" \
      "${BUILD_REQUEST} resolves to ${BUILD_DIR}" \
      "inspect the alias and choose a real /tmp directory"
  if [[ -e "${BUILD_DIR}" ]] && mountpoint -q -- "${BUILD_DIR}"; then
    fail "build clean boundary" "a non-mountpoint directory" "${BUILD_DIR}" \
      "unmount it or choose another dedicated /tmp directory"
  fi
  CLI="${BUILD_DIR}/bin/yolo_defect_cpp"
  IMAGE_PROBE="${BUILD_DIR}/bin/yolo_defect_image_probe"
}

resolve_python() {
  need_command "${PYTHON_REQUEST}" \
    "install Python dependencies or set YOLO_DEFECT_PYTHON"
  PYTHON_EXE="$(command -v -- "${PYTHON_REQUEST}")"
  local output=""
  if ! output="$("${PYTHON_EXE}" -c \
    'import cv2,numpy,onnxruntime as o; assert o.__version__=="1.19.2"; assert "CPUExecutionProvider" in o.get_available_providers(); print(o.__version__,cv2.__version__,numpy.__version__)' \
    2>&1)"; then
    fail "Python dependencies" "cv2, numpy, ORT 1.19.2, CPUExecutionProvider" \
      "${output}" "select the documented Python environment"
  fi
  printf '[pass] Python ORT/OpenCV/NumPy: %s\n' "${output}"
}

resolve_ort() {
  [[ -n "${ORT_REQUEST}" && -d "${ORT_REQUEST}" ]] ||
    fail "ONNXRUNTIME_ROOT" "the official Linux x64 SDK directory" \
      "${ORT_REQUEST:-empty}" "export ONNXRUNTIME_ROOT"
  ORT_ROOT="$(realpath -e -- "${ORT_REQUEST}")"
  local path=""
  for path in VERSION_NUMBER include/onnxruntime_c_api.h \
      include/onnxruntime_cxx_api.h include/cpu_provider_factory.h \
      lib/libonnxruntime.so; do
    need_file "${ORT_ROOT}/${path}" "ORT SDK component" \
      "point ONNXRUNTIME_ROOT at the complete official Linux x64 SDK"
  done
  local version=""
  version="$(tr -d '[:space:]' < "${ORT_ROOT}/VERSION_NUMBER")"
  [[ "${version}" == 1.19.2 ]] ||
    fail "ORT SDK version" "1.19.2" "${version}" "select the pinned SDK"
  local description=""
  description="$(file -L -b -- "${ORT_ROOT}/lib/libonnxruntime.so")"
  [[ "${description}" == *ELF* && "${description}" == *x86-64* ]] ||
    fail "ORT library" "an x86-64 ELF shared object" "${description}" \
      "select the Linux x64 SDK, not Windows or AArch64"
  local elf_header=""
  elf_header="$(readelf -h "${ORT_ROOT}/lib/libonnxruntime.so")"
  grep -q 'Machine:.*X86-64' <<<"${elf_header}" ||
    fail "ORT ELF machine" "X86-64" "unexpected readelf header" \
      "select the Linux x64 SDK"
  export ONNXRUNTIME_ROOT="${ORT_ROOT}"
  export LD_LIBRARY_PATH="${ORT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

resolve_opencv() {
  pkg-config --exists opencv4 ||
    fail "OpenCV C++" "a pkg-config opencv4 package" "not found" \
      "install libopencv-dev"
  OPENCV_VERSION="$(pkg-config --modversion opencv4)"
  [[ "${OPENCV_VERSION}" =~ ^4\.[0-9]+(\.[0-9]+){0,2}$ ]] ||
    fail "OpenCV version" "OpenCV 4.x" "${OPENCV_VERSION}" \
      "select a compatible OpenCV 4 development package"
  OPENCV_CMAKE_DIR="$(realpath -e -- \
    "$(pkg-config --variable=libdir opencv4)/cmake/opencv4")"
  need_file "${OPENCV_CMAKE_DIR}/OpenCVConfig.cmake" \
    "OpenCV CMake package" "install the matching libopencv-dev package"
}

resolve_gtest() {
  if [[ -n "${GTEST_REQUEST}" ]]; then
    [[ -d "${GTEST_REQUEST}" ]] ||
      fail "GoogleTest source" "an existing source directory" \
        "${GTEST_REQUEST}" "correct YOLO_DEFECT_GTEST_SOURCE"
    GTEST_SOURCE="$(realpath -e -- "${GTEST_REQUEST}")"
  elif [[ -f /usr/src/googletest/CMakeLists.txt ]]; then
    GTEST_SOURCE=/usr/src/googletest
  fi
  if [[ -n "${GTEST_SOURCE}" ]]; then
    need_file "${GTEST_SOURCE}/CMakeLists.txt" "GoogleTest source" \
      "set YOLO_DEFECT_GTEST_SOURCE to its source root"
  fi
}

preflight() {
  local tool=""
  for tool in realpath uname file readelf ldd cmake ctest ninja pkg-config \
      cc c++ find mktemp mountpoint grep sed awk; do
    need_command "${tool}" "install the minimal Linux build/debug tools"
  done
  [[ "$(uname -s)" == Linux ]] ||
    fail "host OS" "Linux" "$(uname -s)" "run inside WSL2/Linux"
  [[ "$(uname -m)" == x86_64 ]] ||
    fail "host architecture" "x86_64" "$(uname -m)" \
      "Gate A is native x86_64; AArch64 belongs to Gate B"
  resolve_build_dir
  need_file "${CONFIG}" "default RuntimeConfig" "restore the tracked file"
  need_file "${DEMO_IMAGE}" "fixed demo image" "restore the tracked file"
  need_file "${MANIFEST}" "consistency manifest" "restore the tracked file"
  need_file "${BATCH_MANIFEST}" "S2-03 path-list manifest" \
    "restore the tracked file"
  need_file "${BATCH_VALIDATOR}" "BatchSummary validator" \
    "restore cpp_infer/tools/validate_batch_summary.py"
  need_file "${BATCH_COMPARISON_TOOL}" "batch comparison tool" \
    "restore cpp_infer/tools/compare_batch_runs.py"
  resolve_opencv
  resolve_ort
  resolve_python
  resolve_gtest
}

doctor() {
  stage "read-only Linux x86_64 doctor"
  local host_kind="Linux"
  grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null && host_kind="WSL/Linux"
  printf '[pass] host:       %s %s, kernel=%s\n' \
    "${host_kind}" "$(uname -m)" "$(uname -r)"
  printf '[pass] compiler:   %s / %s\n' "$(command -v cc)" "$(command -v c++)"
  printf '[pass] CMake:      %s; Ninja %s\n' \
    "$(cmake --version | sed -n '1p')" "$(ninja --version)"
  printf '[pass] ORT SDK:    %s (1.19.2)\n' "${ORT_ROOT}"
  printf '[pass] OpenCV:     %s (pkg-config opencv4)\n' "${OPENCV_VERSION}"
  printf '[pass] Python:     %s\n' "${PYTHON_EXE}"
  if [[ -n "${GTEST_SOURCE}" ]]; then
    printf '[pass] GoogleTest: %s\n' "${GTEST_SOURCE}"
  else
    printf '[warn] GoogleTest source missing; a new configure will stop\n'
  fi
  printf '[pass] build:      %s\n' "${BUILD_DIR}"
  printf '[pass] defaults:   config=%s; image=%s; warmup=10; repeat=100\n' \
    "${CONFIG}" "${DEMO_IMAGE}"
  printf '[pass] doctor created no build or evidence\n'
}

check_outputs() {
  [[ -x "${CLI}" ]] ||
    fail "CLI output" "${CLI}" "missing" "inspect the Release build"
  [[ -x "${IMAGE_PROBE}" ]] ||
    fail "image probe output" "${IMAGE_PROBE}" "missing" \
      "configure with BUILD_TESTING=ON"
}

check_ldd() {
  stage "ELF/ldd dependency check"
  local exe="" description="" deps=""
  local count=0 cli_seen=0
  while IFS= read -r -d '' exe; do
    description="$(file -b -- "${exe}")"
    [[ "${description}" == *ELF* && "${description}" == *executable* ]] || continue
    count=$((count + 1))
    printf '[ldd] %s\n' "${exe}"
    deps="$(env -u LD_LIBRARY_PATH ldd "${exe}" 2>&1)" ||
      fail "ldd ${exe}" "a readable dynamic dependency list" "${deps}" \
        "inspect the ELF loader and RPATH"
    ! grep -Eq '=>[[:space:]]+not found' <<<"${deps}" ||
      fail "dependencies for ${exe}" "no not found entries" "${deps}" \
        "install the missing library or correct the SDK/RPATH"
    if [[ "${exe}" == "${CLI}" ]]; then
      cli_seen=1
      grep -q 'libonnxruntime.so' <<<"${deps}" ||
        fail "CLI ORT dependency" "resolved libonnxruntime.so" "${deps}" \
          "link the CLI to the Linux ORT SDK"
      grep -Fq "${ORT_ROOT}/lib/" <<<"${deps}" ||
        fail "CLI ORT RPATH" "libonnxruntime resolved below ${ORT_ROOT}/lib" \
          "${deps}" "configure the Linux BUILD_RPATH from ONNXRUNTIME_ROOT"
      grep -q 'libopencv_' <<<"${deps}" ||
        fail "CLI OpenCV dependency" "resolved libopencv_*" "${deps}" \
          "link the Runtime to OpenCV"
      printf '%s\n' "${deps}" |
        grep -E 'libonnxruntime\.so|libopencv_(core|imgcodecs|imgproc)\.so' |
        sed 's/^/  /'
    fi
    printf '[pass] %s: all dynamic dependencies resolved\n' "$(basename -- "${exe}")"
  done < <(find "${BUILD_DIR}/bin" -maxdepth 1 -type f \
    -name 'yolo_defect_*' -perm -u+x -print0)
  (( count > 0 && cli_seen == 1 )) ||
    fail "ELF inventory" "the CLI and built test executables" "incomplete" \
      "inspect ${BUILD_DIR}/bin"
  printf '[pass] %d ELF executable(s); no dependency is not found\n' "${count}"
}

configure_build() {
  [[ -n "${GTEST_SOURCE}" ]] ||
    fail "GoogleTest source" "a local source directory" "not configured" \
      "install libgtest-dev or set YOLO_DEFECT_GTEST_SOURCE"
  local -a args=(
    -S "${CPP_DIR}" -B "${BUILD_DIR}" -G Ninja
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
    "-DONNXRUNTIME_ROOT=${ORT_ROOT}"
    "-DPython3_EXECUTABLE=${PYTHON_EXE}"
    "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${GTEST_SOURCE}"
    "-DOpenCV_DIR=${OPENCV_CMAKE_DIR}"
  )
  run "CMake configure" cmake "${args[@]}"
  run "Release build" cmake --build "${BUILD_DIR}" --parallel
  check_outputs
  check_ldd
}

clean_build() {
  stage "clean Ninja Release build"
  if [[ -e "${BUILD_DIR}" ]]; then
    [[ ! -L "${BUILD_DIR}" ]] ||
      fail "build clean boundary" "a real directory" "symbolic link" \
        "remove it manually after inspection"
    printf '[clean] %s\n' "${BUILD_DIR}"
    rm -rf -- "${BUILD_DIR}"
  fi
  configure_build
}

ensure_build() {
  if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    stage "incremental Release build"
    configure_build
  else
    clean_build
  fi
}

new_run_dir() {
  if [[ -n "${RUN_REQUEST}" ]]; then
    RUN_DIR="$(absolute_output "${RUN_REQUEST}")"
    [[ "${RUN_DIR}" != / && "${RUN_DIR}" != "${REPO_ROOT}" &&
       "${RUN_DIR}" != "${CPP_DIR}" ]] ||
      fail "result root" "a dedicated directory" "${RUN_DIR}" \
        "set YOLO_DEFECT_RUN_DIR to a Gate-specific output directory"
    [[ ! -e "${RUN_DIR}" || -d "${RUN_DIR}" ]] ||
      fail "result root" "a directory path" "${RUN_DIR}" \
        "choose a fresh Gate-specific output directory"
    local section=""
    for section in demo consistency benchmark batch batch_workers_1 \
        batch_workers_4 batch_comparison.json; do
      [[ ! -e "${RUN_DIR}/${section}" ]] ||
        fail "result root" "no existing ${section} result" \
          "${RUN_DIR}/${section}" \
          "choose a fresh run directory so earlier results stay immutable"
    done
    mkdir -p -- "${RUN_DIR}"
  else
    mkdir -p -- "${BUILD_DIR}/stage1_evidence"
    RUN_DIR="$(mktemp -d "${BUILD_DIR}/stage1_evidence/$(date -u +%Y%m%d_%H%M%S)_XXXXXXXX")"
  fi
}

check_json() {
  [[ -s "$1" ]] || fail "$2" "a non-empty JSON file" "$1" \
    "inspect the producing command"
  if ! "${PYTHON_EXE}" -m json.tool "$1" >/dev/null; then
    fail "$2" "valid UTF-8 JSON" "$1" "inspect JSON serialization"
  fi
}

run_tests() {
  stage "complete CTest gate"
  local inventory=""
  if ! inventory="$(ctest --test-dir "${BUILD_DIR}" -N 2>&1)"; then
    fail "CTest inventory" "exit code 0" "${inventory}" \
      "inspect the configured Release test tree"
  fi
  printf '%s\n' "${inventory}"
  grep -Eq 'Total Tests: [1-9][0-9]*' <<<"${inventory}" ||
    fail "CTest inventory" "at least one registered test" "${inventory}" \
      "rerun clean-build with BUILD_TESTING=ON"
  run "complete CTest" ctest --test-dir "${BUILD_DIR}" --output-on-failure
}

run_demo() {
  stage "fixed single-image demo"
  local json="${RUN_DIR}/demo/crazing_241.json"
  local png="${RUN_DIR}/demo/crazing_241.png"
  run "demo CLI" "${CLI}" --config "${CONFIG}" --image "${DEMO_IMAGE}" \
    --output-json "${json}" --output-image "${png}"
  check_json "${json}" "demo JSON"
  [[ -s "${png}" ]] || fail "demo PNG" "a non-empty image" "${png}" \
    "inspect OpenCV encoding"
  run "demo JSON validator" "${PYTHON_EXE}" "${DETECTION_VALIDATOR}" \
    "${json}" --expected-image "${DEMO_IMAGE}" --expected-config "${CONFIG}"
  run "demo PNG probe" "${IMAGE_PROBE}" "${png}"
  run "demo detection count" "${PYTHON_EXE}" -c \
    'import json,sys; assert len(json.load(open(sys.argv[1],encoding="utf-8"))["detections"])==3' \
    "${json}"
  printf '[pass] Demo: JSON=%s; PNG=%s\n' "${json}" "${png}"
}

run_consistency() {
  stage "Python ORT versus C++ ORT consistency"
  local out="${RUN_DIR}/consistency"
  run "30-image consistency" "${PYTHON_EXE}" "${CONSISTENCY_TOOL}" \
    --manifest "${MANIFEST}" --cpp-cli "${CLI}" --output-dir "${out}" \
    --cpp-opencv-version "${OPENCV_VERSION}"
  check_json "${out}/per_image.json" "consistency per_image.json"
  check_json "${out}/summary.json" "consistency summary.json"
  run "frozen consistency gate" "${PYTHON_EXE}" -c \
    'import json,sys; s=json.load(open(sys.argv[1],encoding="utf-8")); p=json.load(open(sys.argv[2],encoding="utf-8")); r=s["result"]; assert s["passed"] and r["images_total"]==r["images_passed"]==30; assert r["python_detections_total"]==r["cpp_detections_total"]==r["matched_detections_total"]==62; assert r["max_confidence_abs_error"]<=1e-4 and r["max_bbox_coordinate_abs_error_pixels"]<=1e-2 and r["min_matching_iou"]>=0.999; assert len(s["source_class_results"])==6 and all(x["images_total"]==x["images_passed"]==5 for x in s["source_class_results"]); assert len(p["images"])==30 and all(x["passed"] for x in p["images"])' \
    "${out}/summary.json" "${out}/per_image.json"
  printf '[pass] Consistency: summary=%s\n' "${out}/summary.json"
}

run_benchmark() {
  local warmup="$1" repeat="$2"
  stage "Release benchmark (warmup=${warmup}, repeat=${repeat})"
  local json="${RUN_DIR}/benchmark/yolov8_neu_det_cpu_release.json"
  run "C++ benchmark" "${CLI}" --config "${CONFIG}" --image "${DEMO_IMAGE}" \
    --benchmark --warmup "${warmup}" --repeat "${repeat}" \
    --benchmark-json "${json}"
  check_json "${json}" "benchmark JSON"
  run "benchmark validator" "${PYTHON_EXE}" "${BENCHMARK_VALIDATOR}" \
    "${json}" --expected-image "${DEMO_IMAGE}" \
    --expected-warmup "${warmup}" --expected-repeat "${repeat}"
  printf '[pass] Benchmark: JSON=%s\n' "${json}"
}

run_detect() {
  local image="$1" out="$2" config="$3" overwrite="$4"
  image="$(absolute_file "${image}" "detect image" "pass an existing image")"
  [[ -z "${config}" ]] || config="$(absolute_file "${config}" \
    "detect config" "pass an existing RuntimeConfig")"
  [[ -n "${config}" ]] || config="${CONFIG}"
  local name="$(basename -- "${image}")"
  local stem="${name%.*}"
  [[ -n "${stem}" ]] || stem=image
  if [[ -n "${out}" ]]; then
    out="$(absolute_output "${out}")"
    [[ ! -f "${out}" ]] || fail "detect output" "a directory path" \
      "regular file ${out}" "choose another path"
  else
    mkdir -p -- "${DETECT_ROOT}"
    local safe="$(sed 's/[^A-Za-z0-9._-]/_/g' <<<"${stem}")"
    out="$(mktemp -d "${DETECT_ROOT}/$(date -u +%Y%m%d_%H%M%S)_${safe:-image}_XXXXXXXX")"
  fi
  local json="${out}/${stem}.detections.json"
  local png="${out}/${stem}.visualized.png"
  local -a args=(--config "${config}" --image "${image}"
    --output-json "${json}" --output-image "${png}")
  (( overwrite == 0 )) || args+=(--overwrite)
  stage "arbitrary single-image detection"
  run "detect CLI" "${CLI}" "${args[@]}"
  check_json "${json}" "detect JSON"
  run "detect JSON validator" "${PYTHON_EXE}" "${DETECTION_VALIDATOR}" \
    "${json}" --expected-image "${image}" --expected-config "${config}"
  [[ -s "${png}" ]] || fail "detect PNG" "a non-empty image" "${png}" \
    "inspect OpenCV encoding"
  run "detect PNG probe" "${IMAGE_PROBE}" "${png}"
  printf '[pass] Detect: JSON=%s; PNG=%s\n' "${json}" "${png}"
}

run_batch() {
  local input="$1" out="$2" config="$3" workers="$4" capacity="$5"
  local output_images="$6" overwrite="$7"
  local input_option="" input_kind=""
  [[ "${input}" == /* ]] || input="${CALLER_DIR}/${input}"
  if [[ -d "${input}" ]]; then
    input="$(realpath -e -- "${input}")"
    input_option=--input-dir
    input_kind=directory
  elif [[ -f "${input}" ]]; then
    input="$(realpath -e -- "${input}")"
    input_option=--manifest
    input_kind=manifest
  else
    fail "batch input" "an existing directory or manifest file" "${input}" \
      "pass a directory or UTF-8 path-list manifest"
  fi
  [[ -z "${config}" ]] || config="$(absolute_file "${config}" \
    "batch config" "pass an existing RuntimeConfig")"
  [[ -n "${config}" ]] || config="${CONFIG}"
  if [[ -n "${out}" ]]; then
    out="$(absolute_output "${out}")"
    [[ ! -f "${out}" ]] || fail "batch output" "a directory path" \
      "regular file ${out}" "choose another path"
  else
    mkdir -p -- "${DETECT_ROOT}"
    out="$(mktemp -d "${DETECT_ROOT}/$(date -u +%Y%m%d_%H%M%S)_batch_XXXXXXXX")"
  fi
  local summary="${out}/batch_summary.json"
  local -a args=(--config "${config}" --batch "${input_option}" "${input}"
    --output-dir "${out}" --batch-summary "${summary}"
    --workers "${workers}" --queue-capacity "${capacity}")
  (( output_images == 0 )) || args+=(--output-images)
  (( overwrite == 0 )) || args+=(--overwrite)
  stage "bounded multi-image batch (workers=${workers}, queue=${capacity})"
  run "batch CLI" "${CLI}" "${args[@]}"
  check_json "${summary}" "BatchSummary"
  run "BatchSummary validator" "${PYTHON_EXE}" "${BATCH_VALIDATOR}" \
    "${summary}" --expected-status succeeded \
    --expected-input-kind "${input_kind}" \
    --expected-requested-workers "${workers}"
  printf '[pass] Batch: summary=%s; output=%s\n' "${summary}" "${out}"
}

run_batch_comparison() {
  local config="$1"
  [[ -z "${config}" ]] || config="$(absolute_file "${config}" \
    "batch comparison config" "pass an existing RuntimeConfig")"
  [[ -n "${config}" ]] || config="${CONFIG}"
  stage "formal workers=1 versus workers=4 comparison"
  [[ -d "${BATCH_PERFORMANCE_INPUT}" ]] ||
    fail "batch performance input" "data/images/val directory" \
      "${BATCH_PERFORMANCE_INPUT}" "restore the validation images"
  local workers_1_out="${RUN_DIR}/batch_workers_1"
  local workers_4_out="${RUN_DIR}/batch_workers_4"
  run_batch "${BATCH_PERFORMANCE_INPUT}" "${workers_1_out}" "${config}" \
    1 8 0 0
  run_batch "${BATCH_PERFORMANCE_INPUT}" "${workers_4_out}" "${config}" \
    4 8 0 0
  local comparison="${RUN_DIR}/batch_comparison.json"
  run "batch comparison" "${PYTHON_EXE}" "${BATCH_COMPARISON_TOOL}" \
    --workers-1-summary "${workers_1_out}/batch_summary.json" \
    --workers-4-summary "${workers_4_out}/batch_summary.json" \
    --output "${comparison}"
  check_json "${comparison}" "batch comparison JSON"
  printf '[pass] Batch comparison: %s\n' "${comparison}"
}

main() {
  local action="${1:-help}"
  (( $# == 0 )) || shift
  local detect_image="" detect_out="" detect_config="" detect_overwrite=0
  local batch_input="" batch_out="" batch_config="" batch_workers=1
  local batch_capacity="" batch_output_images=0 batch_overwrite=0
  local batch_comparison_config=""
  local warmup=10 repeat=100

  case "${action}" in
    help)
      (( $# == 0 )) || fail "help arguments" "none" "$*" "remove them"
      usage
      return
      ;;
    detect)
      (( $# > 0 )) || fail "detect image" "one path" "missing" \
        "run stage1.sh detect <image> [output-dir]"
      detect_image="$1"
      shift
      if (( $# > 0 )) && [[ "$1" != --* ]]; then detect_out="$1"; shift; fi
      while (( $# > 0 )); do
        case "$1" in
          --config) (( $# > 1 )) || fail "--config" "a value" "missing" \
            "pass a RuntimeConfig"; detect_config="$2"; shift 2 ;;
          --overwrite) detect_overwrite=1; shift ;;
          *) fail "detect argument" "--config or --overwrite" "$1" \
            "run stage1.sh help" ;;
        esac
      done
      ;;
    batch)
      (( $# > 0 )) || fail "batch input" "one directory or manifest path" \
        "missing" "run stage1.sh batch <input-dir-or-manifest> [output-dir]"
      batch_input="$1"
      shift
      if (( $# > 0 )) && [[ "$1" != --* ]]; then batch_out="$1"; shift; fi
      while (( $# > 0 )); do
        case "$1" in
          --config) (( $# > 1 )) || fail "--config" "a value" "missing" \
            "pass a RuntimeConfig"; batch_config="$2"; shift 2 ;;
          --workers) (( $# > 1 )) || fail "--workers" "a value" "missing" \
            "pass an integer"; batch_workers="$2"; shift 2 ;;
          --queue-capacity) (( $# > 1 )) || fail "--queue-capacity" \
            "a value" "missing" "pass an integer"; batch_capacity="$2"; shift 2 ;;
          --output-images) batch_output_images=1; shift ;;
          --overwrite) batch_overwrite=1; shift ;;
          *) fail "batch argument" "--config/--workers/--queue-capacity/--output-images/--overwrite" \
            "$1" "run stage1.sh help" ;;
        esac
      done
      [[ "${batch_workers}" =~ ^[0-9]+$ ]] &&
        (( batch_workers >= 1 && batch_workers <= 64 )) ||
        fail "batch workers" "an integer in [1,64]" "${batch_workers}" \
          "correct --workers"
      if [[ -z "${batch_capacity}" ]]; then
        batch_capacity=$((2 * batch_workers))
      fi
      [[ "${batch_capacity}" =~ ^[0-9]+$ ]] &&
        (( batch_capacity >= 1 && batch_capacity <= 4096 )) ||
        fail "batch queue capacity" "an integer in [1,4096]" \
          "${batch_capacity}" "correct --queue-capacity"
      ;;
    batch-compare)
      while (( $# > 0 )); do
        case "$1" in
          --config) (( $# > 1 )) || fail "--config" "a value" "missing" \
            "pass a RuntimeConfig"; batch_comparison_config="$2"; shift 2 ;;
          *) fail "batch-compare argument" "--config <path>" "$1" \
            "the protocol fixes input/workers/queue/output policy" ;;
        esac
      done
      ;;
    benchmark)
      while (( $# > 0 )); do
        case "$1" in
          --warmup) (( $# > 1 )) || fail "--warmup" "a value" "missing" \
            "pass an integer"; warmup="$2"; shift 2 ;;
          --repeat) (( $# > 1 )) || fail "--repeat" "a value" "missing" \
            "pass an integer"; repeat="$2"; shift 2 ;;
          *) fail "benchmark argument" "--warmup or --repeat" "$1" \
            "run stage1.sh help" ;;
        esac
      done
      [[ "${warmup}" =~ ^[0-9]+$ && "${repeat}" =~ ^[0-9]+$ ]] ||
        fail "benchmark counts" "non-negative integers" "${warmup}/${repeat}" \
          "correct --warmup/--repeat"
      (( warmup <= 100000 && repeat >= 1 && repeat <= 100000 )) ||
        fail "benchmark counts" "warmup [0,100000], repeat [1,100000]" \
          "${warmup}/${repeat}" "correct the counts"
      ;;
    doctor|build|clean-build|test|demo|consistency|all)
      (( $# == 0 )) || fail "${action} arguments" "none" "$*" "remove them"
      ;;
    *) fail "workflow action" \
      "help/doctor/build/clean-build/test/detect/batch/batch-compare/demo/consistency/benchmark/all" \
      "${action}" "run stage1.sh help" ;;
  esac

  preflight
  printf '[env] build=%s\n[env] ORT=%s\n[env] OpenCV=%s\n[env] Python=%s\n' \
    "${BUILD_DIR}" "${ORT_ROOT}" "${OPENCV_VERSION}" "${PYTHON_EXE}"
  case "${action}" in
    doctor) doctor ;;
    build) ensure_build ;;
    clean-build) clean_build ;;
    test) ensure_build; run_tests ;;
    detect) ensure_build; run_detect "${detect_image}" "${detect_out}" \
      "${detect_config}" "${detect_overwrite}" ;;
    batch) ensure_build; run_batch "${batch_input}" "${batch_out}" \
      "${batch_config}" "${batch_workers}" "${batch_capacity}" \
      "${batch_output_images}" "${batch_overwrite}" ;;
    batch-compare) ensure_build; new_run_dir; \
      run_batch_comparison "${batch_comparison_config}" ;;
    demo) ensure_build; new_run_dir; run_demo ;;
    consistency) ensure_build; new_run_dir; run_consistency ;;
    benchmark) ensure_build; new_run_dir; run_consistency; \
      run_benchmark "${warmup}" "${repeat}" ;;
    all) clean_build; run_tests; new_run_dir; run_demo; run_consistency; \
      run_benchmark 10 100; run_batch "${BATCH_MANIFEST}" \
      "${RUN_DIR}/batch" "${CONFIG}" 2 4 0 0 ;;
  esac
  stage "${action} PASS"
  printf 'Build directory: %s\n' "${BUILD_DIR}"
  [[ -z "${RUN_DIR}" ]] || printf 'Fresh evidence:  %s\n' "${RUN_DIR}"
}

main "$@"
