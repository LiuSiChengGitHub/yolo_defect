#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CPP_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${CPP_DIR}/.." && pwd -P)"
TOOLCHAIN="${CPP_DIR}/cmake/toolchains/linux-aarch64-gnu.cmake"
DEFAULT_CONFIG="${CPP_DIR}/configs/default_config.txt"
CONFIG_REQUEST="${YOLO_DEFECT_AARCH64_CONFIG:-${DEFAULT_CONFIG}}"
CONFIG=""
DEMO_IMAGE="${REPO_ROOT}/data/images/val/crazing_241.jpg"
SECOND_BATCH_IMAGE="${REPO_ROOT}/data/images/val/crazing_242.jpg"
DETECTION_VALIDATOR="${CPP_DIR}/tests/assert_detection_json.py"
BATCH_SUMMARY_VALIDATOR="${CPP_DIR}/tools/validate_batch_summary.py"

DEPS_ROOT="${YOLO_DEFECT_AARCH64_DEPS_ROOT:-${HOME}/.local/opt/yolo-defect-aarch64}"
ORT_ROOT="${YOLO_DEFECT_AARCH64_ORT_ROOT:-${DEPS_ROOT}/onnxruntime-linux-aarch64-1.19.2}"
TARGET_SYSROOT="${YOLO_DEFECT_AARCH64_SYSROOT:-${DEPS_ROOT}/ubuntu-noble-opencv-4.6.0}"
LOADER_PREFIX="${YOLO_DEFECT_AARCH64_LOADER_PREFIX:-/usr/aarch64-linux-gnu}"
CORE_BUILD="${YOLO_DEFECT_AARCH64_CORE_BUILD_DIR:-/tmp/yolo_defect_stage2_aarch64_core}"
FULL_BUILD="${YOLO_DEFECT_AARCH64_FULL_BUILD_DIR:-/tmp/yolo_defect_stage2_aarch64_full}"
RESULT_DIR="${YOLO_DEFECT_AARCH64_RESULT_DIR:-${CPP_DIR}/results/s2_02/aarch64_qemu}"
BATCH_RESULT_DIR="${YOLO_DEFECT_AARCH64_BATCH_RESULT_DIR:-${CPP_DIR}/results/s2_03/linux_aarch64_qemu}"

CORE_SMOKE="${CORE_BUILD}/bin/yolo_defect_project_core_smoke"
RUNTIME_OBJECT="${FULL_BUILD}/CMakeFiles/yolo_defect_runtime.dir/src/detector_pipeline.cpp.o"
DEPLOY_DIR="${FULL_BUILD}/deploy"
CLI="${DEPLOY_DIR}/bin/yolo_defect_cpp"
TARGET_LIBRARY_PATH="${DEPLOY_DIR}/lib:${TARGET_SYSROOT}/usr/lib/aarch64-linux-gnu:${TARGET_SYSROOT}/lib/aarch64-linux-gnu:${TARGET_SYSROOT}/usr/lib/aarch64-linux-gnu/blas:${TARGET_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack:${TARGET_SYSROOT}/usr/lib:${LOADER_PREFIX}/lib"
HOST_KERNEL_ARCHITECTURE="$(uname -m)"

fail() {
  printf '%s\n' \
    "Gate B failed: object=$1; expected=$2; actual=$3; action=$4" >&2
  exit 1
}

stage() {
  printf '\n[Gate B AArch64] %s\n' "$1"
}

need_command() {
  command -v -- "$1" >/dev/null 2>&1 ||
    fail "command $1" "available on the x86_64 host PATH" "not found" "$2"
}

need_file() {
  [[ -f "$1" ]] || fail "$2" "an existing regular file" "$1" "$3"
}

usage() {
  cat <<'EOF'
YOLO Defect Linux x86_64 -> Linux AArch64 Gate B workflow

Usage:
  stage2_aarch64.sh help
  stage2_aarch64.sh doctor
  stage2_aarch64.sh build
  stage2_aarch64.sh clean-build
  stage2_aarch64.sh inspect
  stage2_aarch64.sh smoke
  stage2_aarch64.sh infer
  stage2_aarch64.sh batch
  stage2_aarch64.sh all

Actions:
  doctor       Check host tools, target compiler/loader, ARM64 ORT/OpenCV.
  build        Cross-compile project-core, Runtime, and production CLI.
  clean-build  Recreate two guarded /tmp Ninja Release build trees.
  inspect      Prove AArch64 ELF/interpreter/NEEDED and resolve target .so files.
  smoke        QEMU core logic, CLI startup/help, config/artifact, error paths.
  infer        Fixed image -> ARM64 ORT CPU -> Detection JSON under QEMU.
  batch        QEMU directory/manifest bounded-concurrency and partial-failure acceptance.
  all          Doctor -> clean build -> inspect -> smoke -> infer -> batch.

The batch action records functional evidence under
cpp_infer/results/s2_03/linux_aarch64_qemu by default. No action records or
interprets QEMU latency, throughput, memory, power, or board performance.
Set YOLO_DEFECT_AARCH64_CONFIG to select another existing RuntimeConfig;
the default remains configs/default_config.txt (FP32).
Run bootstrap_aarch64_deps.sh first; see docs/paths_commands.md.
EOF
}

guard_build_dir() {
  local requested="$1"
  local expected_leaf="$2"
  local lexical=""
  local resolved=""
  lexical="$(realpath -m -s -- "${requested}")"
  resolved="$(realpath -m -- "${requested}")"
  [[ "${resolved}" == /tmp/* && "${resolved##*/}" == "${expected_leaf}" ]] ||
    fail "clean boundary" "/tmp/${expected_leaf}" "${resolved}" \
      "use the documented dedicated build directory"
  [[ "${lexical}" == "${resolved}" ]] ||
    fail "clean boundary" "a path without symlink components" \
      "${requested} resolves to ${resolved}" "choose a real /tmp directory"
  if [[ -e "${resolved}" ]] && mountpoint -q -- "${resolved}"; then
    fail "clean boundary" "a non-mountpoint directory" "${resolved}" \
      "unmount it or choose the documented /tmp directory"
  fi
}

verify_aarch64_elf() {
  local path="$1"
  local label="$2"
  local description=""
  need_file "${path}" "${label}" "build or fetch the target artifact first"
  description="$(file -L -b -- "${path}")"
  [[ "${description}" == *ELF* && "${description}" == *ARM\ aarch64* ]] ||
    fail "${label} architecture" "ARM aarch64 ELF" "${description}" \
      "remove the mixed-architecture build/dependency and rebuild"
  aarch64-linux-gnu-readelf -h "${path}" | grep -q 'Machine:.*AArch64' ||
    fail "${label} ELF machine" "AArch64" "unexpected readelf header" \
      "select target AArch64 rather than host x86_64 inputs"
}

doctor() {
  stage "host tools and target dependencies"
  [[ -f "${CONFIG_REQUEST}" ]] ||
    fail "RuntimeConfig" "an existing regular file" "${CONFIG_REQUEST}" \
      "set YOLO_DEFECT_AARCH64_CONFIG to the selected config"
  CONFIG="$(realpath -e -- "${CONFIG_REQUEST}")"
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
    fail "host" "Linux x86_64" "$(uname -s)/$(uname -m)" \
      "run inside the documented WSL2 Ubuntu x86_64 host"
  for command_name in cmake ninja aarch64-linux-gnu-g++ \
      aarch64-linux-gnu-readelf cmp file qemu-aarch64 python3 readelf timeout; do
    need_command "${command_name}" "install the documented minimal host tools"
  done
  printf '[pass] selected RuntimeConfig: %s\n' "${CONFIG}"
  [[ "$(aarch64-linux-gnu-g++ -dumpmachine)" == aarch64-linux-gnu ]] ||
    fail "compiler target" "aarch64-linux-gnu" \
      "$(aarch64-linux-gnu-g++ -dumpmachine)" \
      "select the Ubuntu AArch64 GNU cross compiler"
  need_file "${TOOLCHAIN}" "CMake toolchain" "restore the repository file"
  need_file "${ORT_ROOT}/VERSION_NUMBER" "ARM64 ORT SDK" \
    "run bootstrap_aarch64_deps.sh fetch"
  [[ "$(tr -d '[:space:]' < "${ORT_ROOT}/VERSION_NUMBER")" == 1.19.2 ]] ||
    fail "ARM64 ORT version" "1.19.2" \
      "$(tr -d '[:space:]' < "${ORT_ROOT}/VERSION_NUMBER")" \
      "select the pinned official SDK"
  need_file "${LOADER_PREFIX}/lib/ld-linux-aarch64.so.1" \
    "target dynamic loader" "install libc6-dev-arm64-cross"
  verify_aarch64_elf "${LOADER_PREFIX}/lib/ld-linux-aarch64.so.1" \
    "target dynamic loader"
  verify_aarch64_elf "${ORT_ROOT}/lib/libonnxruntime.so.1.19.2" \
    "ORT target library"
  for component in core imgproc imgcodecs; do
    verify_aarch64_elf \
      "${TARGET_SYSROOT}/usr/lib/aarch64-linux-gnu/libopencv_${component}.so.4.6.0" \
      "OpenCV ${component} target library"
  done
  need_file "${TARGET_SYSROOT}/usr/include/opencv4/opencv2/core/version.hpp" \
    "target OpenCV headers" "run bootstrap_aarch64_deps.sh fetch"
  printf '[pass] host=%s; target=%s; compiler=%s; qemu=%s\n' \
    "$(uname -m)" "$(aarch64-linux-gnu-g++ -dumpmachine)" \
    "$(aarch64-linux-gnu-g++ -dumpversion)" \
    "$(qemu-aarch64 --version | sed -n '1p')"
  printf '[pass] ORT_ROOT=%s\n' "${ORT_ROOT}"
  printf '[pass] TARGET_SYSROOT=%s\n' "${TARGET_SYSROOT}"
}

configure_and_build() {
  stage "cross-compile dependency-free project core"
  cmake -S "${CPP_DIR}" -B "${CORE_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DYOLO_DEFECT_CORE_ONLY=ON \
    -DBUILD_TESTING=OFF
  cmake --build "${CORE_BUILD}" --target \
    yolo_defect_project_core yolo_defect_project_core_smoke

  stage "cross-compile shared-source Runtime and production CLI"
  cmake -S "${CPP_DIR}" -B "${FULL_BUILD}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DYOLO_DEFECT_AARCH64_SYSROOT="${TARGET_SYSROOT}" \
    -DONNXRUNTIME_ROOT="${ORT_ROOT}" \
    -DBUILD_TESTING=OFF
  cmake --build "${FULL_BUILD}" --target \
    yolo_defect_runtime yolo_defect_cpp

  mkdir -p -- "${DEPLOY_DIR}/bin" "${DEPLOY_DIR}/lib"
  cmake -E copy "${FULL_BUILD}/bin/yolo_defect_cpp" "${CLI}"
  cp -a -- "${ORT_ROOT}"/lib/libonnxruntime.so* "${DEPLOY_DIR}/lib/"
  printf '[pass] deploy directory=%s (CLI + ORT; Ubuntu target libs stay in sysroot)\n' \
    "${DEPLOY_DIR}"
}

clean_build() {
  guard_build_dir "${CORE_BUILD}" yolo_defect_stage2_aarch64_core
  guard_build_dir "${FULL_BUILD}" yolo_defect_stage2_aarch64_full
  rm -rf -- "${CORE_BUILD}" "${FULL_BUILD}"
  configure_and_build
}

need_build() {
  need_file "${CORE_SMOKE}" "project-core smoke ELF" \
    "run stage2_aarch64.sh build"
  need_file "${FULL_BUILD}/libyolo_defect_runtime.a" "Runtime archive" \
    "run stage2_aarch64.sh build"
  need_file "${RUNTIME_OBJECT}" "Runtime AArch64 object" \
    "run stage2_aarch64.sh build"
  need_file "${CLI}" "deployed production CLI" \
    "run stage2_aarch64.sh build"
}

inspect_artifacts() {
  need_build
  stage "AArch64 static and dynamic proof"
  mkdir -p -- "${RESULT_DIR}"
  local evidence="${RESULT_DIR}/elf_inspection.txt"
  {
    printf 'host=%s/%s\n' "$(uname -s)" "$(uname -m)"
    printf 'compiler_target=%s\n' "$(aarch64-linux-gnu-g++ -dumpmachine)"
    printf '\n[file]\n'
    file -L -- "${CORE_SMOKE}" "${RUNTIME_OBJECT}" "${CLI}" \
      "${ORT_ROOT}/lib/libonnxruntime.so.1.19.2"
    printf '\n[cli ELF header]\n'
    aarch64-linux-gnu-readelf -h "${CLI}"
    printf '\n[cli program interpreter]\n'
    aarch64-linux-gnu-readelf -l "${CLI}" | grep -E \
      'Requesting program interpreter|INTERP'
    printf '\n[cli dynamic dependencies]\n'
    aarch64-linux-gnu-readelf -d "${CLI}" | grep -E \
      'NEEDED|RPATH|RUNPATH'
  } | tee "${evidence}"

  verify_aarch64_elf "${CORE_SMOKE}" "project-core smoke"
  verify_aarch64_elf "${RUNTIME_OBJECT}" "Runtime object"
  verify_aarch64_elf "${CLI}" "production CLI"
  aarch64-linux-gnu-readelf -l "${CLI}" |
    grep -q '/lib/ld-linux-aarch64.so.1' ||
    fail "CLI interpreter" "/lib/ld-linux-aarch64.so.1" "unexpected" \
      "rebuild with the documented AArch64 GNU toolchain"

  local loader_output="${RESULT_DIR}/loader_resolution.txt"
  qemu-aarch64 -L "${LOADER_PREFIX}" \
    -E "LD_LIBRARY_PATH=${TARGET_LIBRARY_PATH}" \
    "${LOADER_PREFIX}/lib/ld-linux-aarch64.so.1" --list "${CLI}" |
    tee "${loader_output}"
  ! grep -q 'not found' "${loader_output}" ||
    fail "target dynamic dependencies" "all resolved" "not found entry" \
      "inspect loader_resolution.txt and add the missing ARM64 package"

  local resolved_library=""
  local -a resolved_libraries=()
  mapfile -t resolved_libraries < <(
    awk '$2 == "=>" && $3 ~ /^\// {print $3}' "${loader_output}" | sort -u)
  ((${#resolved_libraries[@]} > 0)) ||
    fail "target loader proof" "one or more resolved libraries" "none" \
      "inspect QEMU and the target loader command"
  for resolved_library in "${resolved_libraries[@]}"; do
    verify_aarch64_elf "${resolved_library}" "resolved target library"
  done
  printf '[pass] inspected %d resolved target libraries; none are x86_64\n' \
    "${#resolved_libraries[@]}"
}

qemu_target() {
  qemu-aarch64 -L "${LOADER_PREFIX}" \
    -E "LD_LIBRARY_PATH=${TARGET_LIBRARY_PATH}" \
    -E "YOLO_DEFECT_RUNTIME_KERNEL_ARCHITECTURE=${HOST_KERNEL_ARCHITECTURE}" \
    -E "YOLO_DEFECT_EXECUTION_CONTEXT=qemu_user_mode_on_x86_64_host" "$@"
}

qemu_target_timed() {
  timeout 600 qemu-aarch64 -L "${LOADER_PREFIX}" \
    -E "LD_LIBRARY_PATH=${TARGET_LIBRARY_PATH}" \
    -E "YOLO_DEFECT_RUNTIME_KERNEL_ARCHITECTURE=${HOST_KERNEL_ARCHITECTURE}" \
    -E "YOLO_DEFECT_EXECUTION_CONTEXT=qemu_user_mode_on_x86_64_host" "$@"
}

expect_contract_failure() {
  local config_path="$1"
  local label="$2"
  local output=""
  if output="$(qemu_target "${CLI}" --config "${config_path}" 2>&1)"; then
    fail "${label}" "nonzero exit" "exit code 0" \
      "restore the RuntimeConfig/ModelArtifactSpec validation contract"
  fi
  grep -Eqi 'expected|must be' <<<"${output}" ||
    fail "${label} diagnostic" "expected value/rule" "${output}" \
      "preserve an actionable expected/actual/action error"
  grep -Eqi 'actual|got|unknown' <<<"${output}" ||
    fail "${label} diagnostic" "actual offending value" "${output}" \
      "preserve an actionable expected/actual/action error"
  grep -Eqi 'action|fix|remove|set|use' <<<"${output}" ||
    fail "${label} diagnostic" "corrective action" "${output}" \
      "preserve an actionable expected/actual/action error"
  printf '[expected failure] %s\n%s\n' "${label}" "${output}"
  printf '[pass] %s rejected with actionable diagnostic\n' "${label}"
}

run_smokes() {
  need_build
  stage "QEMU correctness smokes (not performance)"
  mkdir -p -- "${RESULT_DIR}"
  local evidence="${RESULT_DIR}/qemu_smoke.txt"
  {
    printf '[project core]\n'
    qemu_target "${CORE_SMOKE}"
    printf '\n[CLI startup]\n'
    qemu_target "${CLI}"
    printf '\n[CLI help]\n'
    qemu_target "${CLI}" --help
    printf '\n[config + artifact contract from /tmp]\n'
    (cd /tmp && qemu_target "${CLI}" --config "${CONFIG}")
    printf '\n[negative contracts]\n'
    expect_contract_failure \
      "${CPP_DIR}/tests/fixtures/runtime/invalid_nms_threshold.txt" \
      "invalid RuntimeConfig threshold"
    expect_contract_failure \
      "${CPP_DIR}/tests/fixtures/runtime/unknown_artifact_field.txt" \
      "unknown ModelArtifactSpec field"
    printf '[pass] QEMU startup/help/contracts/core smoke completed\n'
  } | tee "${evidence}"
  grep -q 'decode -> class-agnostic NMS -> coordinate restore' "${evidence}" ||
    fail "project-core smoke" "real decode/NMS/coordinate restore output" \
      "missing" "restore the existing project-core portability smoke"
  grep -q 'Usage:' "${evidence}" ||
    fail "CLI help" "Usage output" "missing" "restore --help startup behavior"
  grep -q '^model_id: .\+' "${evidence}" ||
    fail "config/artifact contract" "a non-empty selected model_id" \
      "missing" "restore relative artifact resolution and contract validation"
  grep -q 'invalid RuntimeConfig threshold rejected' "${evidence}" ||
    fail "negative RuntimeConfig evidence" "recorded rejection" "missing" \
      "keep negative contract checks inside the raw QEMU evidence"
  grep -q 'unknown ModelArtifactSpec field rejected' "${evidence}" ||
    fail "negative ModelArtifactSpec evidence" "recorded rejection" "missing" \
      "keep negative contract checks inside the raw QEMU evidence"
}

run_full_inference() {
  need_build
  stage "fixed image -> ARM64 ORT CPU -> Detection JSON under QEMU"
  need_file "${DEMO_IMAGE}" "fixed input image" "restore the repository sample"
  need_file "${DETECTION_VALIDATOR}" "detection JSON validator" \
    "restore the repository validator"
  mkdir -p -- "${RESULT_DIR}/detect"
  local output_json="${RESULT_DIR}/detect/crazing_241.detections.json"
  qemu_target_timed "${CLI}" --config "${CONFIG}" --image "${DEMO_IMAGE}" \
    --output-json "${output_json}" --overwrite
  python3 "${DETECTION_VALIDATOR}" "${output_json}" \
    --expected-image "${DEMO_IMAGE}" --expected-config "${CONFIG}"
  local detection_count=""
  detection_count="$(python3 -c \
    'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["detections"]))' \
    "${output_json}")"
  [[ "${detection_count}" == 3 ]] ||
    fail "fixed-image detection count" "3" "${detection_count}" \
      "check ARM64 preprocessing, ORT inference, and postprocessing parity"
  printf '[pass] fixed-image detection_count=3\n'
  printf '[pass] full ARM64 inference executed under QEMU; JSON=%s\n' \
    "${output_json}"
}

validate_batch_summary() {
  local summary_path="$1"
  shift
  python3 "${BATCH_SUMMARY_VALIDATOR}" "${summary_path}" \
    --expected-target-architecture aarch64 \
    --expected-runtime-kernel-architecture "${HOST_KERNEL_ARCHITECTURE}" \
    --expected-execution-context qemu_user_mode_on_x86_64_host \
    --expect-unpublishable-memory "$@"
}

run_batch_acceptance() {
  need_build
  stage "S2-03 bounded multi-image correctness under QEMU (not performance)"
  need_file "${DEMO_IMAGE}" "first fixed batch image" \
    "restore the repository sample"
  need_file "${SECOND_BATCH_IMAGE}" "second fixed batch image" \
    "restore the repository sample"
  need_file "${BATCH_SUMMARY_VALIDATOR}" "BatchSummary validator" \
    "restore cpp_infer/tools/validate_batch_summary.py"

  local run_id="${YOLO_DEFECT_AARCH64_BATCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
  [[ "${run_id}" != */* && "${run_id}" != *\\* && "${run_id}" != "." &&
     "${run_id}" != ".." ]] ||
    fail "AArch64 batch run id" "one safe path component" "${run_id}" \
      "set YOLO_DEFECT_AARCH64_BATCH_RUN_ID without separators"
  local run_root="${BATCH_RESULT_DIR}/${run_id}"
  [[ ! -e "${run_root}" ]] ||
    fail "AArch64 batch run directory" "a fresh non-existing path" \
      "${run_root}" \
      "choose a new YOLO_DEFECT_AARCH64_BATCH_RUN_ID so stale outputs cannot satisfy acceptance"

  local input_root="${run_root}/inputs"
  local good_directory="${input_root}/good_directory"
  local good_manifest="${input_root}/good_manifest.txt"
  local partial_manifest="${input_root}/partial_failure_manifest.txt"
  local corrupt_image="${input_root}/corrupt.jpg"
  local directory_output="${run_root}/directory_workers1"
  local manifest_output="${run_root}/manifest_workers2"
  local partial_output="${run_root}/partial_failure"
  local directory_summary="${directory_output}/batch_summary.json"
  local manifest_summary="${manifest_output}/batch_summary.json"
  local partial_summary="${partial_output}/batch_summary.json"
  local evidence="${run_root}/qemu_batch_acceptance.txt"

  mkdir -p -- "${good_directory}" "${directory_output}" \
    "${manifest_output}" "${partial_output}"
  cp -f -- "${DEMO_IMAGE}" "${good_directory}/crazing_241.jpg"
  cp -f -- "${SECOND_BATCH_IMAGE}" "${good_directory}/crazing_242.jpg"
  printf '%s\n' \
    'good_directory/crazing_241.jpg' \
    'good_directory/crazing_242.jpg' > "${good_manifest}"
  printf '%s\n' 'deliberately corrupt JPEG payload' > "${corrupt_image}"
  printf '%s\n' \
    'good_directory/crazing_241.jpg' \
    'good_directory/crazing_242.jpg' \
    'corrupt.jpg' > "${partial_manifest}"

  {
    printf '[directory: workers=1, queue=2]\n'
    qemu_target_timed "${CLI}" --config "${CONFIG}" --batch \
      --input-dir "${good_directory}" \
      --output-dir "${directory_output}" \
      --batch-summary "${directory_summary}" \
      --workers 1 --queue-capacity 2
    validate_batch_summary "${directory_summary}" \
      --expected-status succeeded \
      --expected-discovered 2 --expected-enqueued 2 --expected-started 2 \
      --expected-succeeded 2 --expected-failed 0 --expected-cancelled 0 \
      --expected-requested-workers 1 --expected-effective-workers 1 \
      --expected-input-kind directory

    printf '\n[manifest: workers=2, queue=1]\n'
    qemu_target_timed "${CLI}" --config "${CONFIG}" --batch \
      --manifest "${good_manifest}" \
      --output-dir "${manifest_output}" \
      --batch-summary "${manifest_summary}" \
      --workers 2 --queue-capacity 1
    validate_batch_summary "${manifest_summary}" \
      --expected-status succeeded \
      --expected-discovered 2 --expected-enqueued 2 --expected-started 2 \
      --expected-succeeded 2 --expected-failed 0 --expected-cancelled 0 \
      --expected-requested-workers 2 --expected-effective-workers 2 \
      --expected-input-kind manifest

    printf '\n[directory/manifest per-image parity]\n'
    local sequence=""
    for sequence in 000000 000001; do
      local directory_json="${directory_output}/items/${sequence}.detections.json"
      local manifest_json="${manifest_output}/items/${sequence}.detections.json"
      cmp -s -- "${directory_json}" "${manifest_json}" ||
        fail "batch detection JSON ${sequence}" \
          "byte-identical directory/manifest output" "different bytes" \
          "check deterministic task order and shared DetectorPipeline semantics"
      python3 -c \
        'import json,sys; assert json.load(open(sys.argv[1], encoding="utf-8")) == json.load(open(sys.argv[2], encoding="utf-8"))' \
        "${directory_json}" "${manifest_json}"
      printf '[pass] %s byte-identical and semantically identical\n' "${sequence}"
    done

    printf '\n[partial failure: exact 2 succeeded + 1 failed]\n'
    local partial_exit=0
    set +e
    qemu_target_timed "${CLI}" --config "${CONFIG}" --batch \
      --manifest "${partial_manifest}" \
      --output-dir "${partial_output}" \
      --batch-summary "${partial_summary}" \
      --workers 2 --queue-capacity 1
    partial_exit=$?
    set -e
    [[ "${partial_exit}" == 2 ]] ||
      fail "partial-failure exit code" "2" "${partial_exit}" \
        "preserve per-image failure isolation and the public batch exit mapping"
    validate_batch_summary "${partial_summary}" \
      --expected-status partial_failure \
      --expected-discovered 3 --expected-enqueued 3 --expected-started 3 \
      --expected-succeeded 2 --expected-failed 1 --expected-cancelled 0 \
      --expected-requested-workers 2 --expected-effective-workers 2 \
      --expected-input-kind manifest
    printf '[pass] corrupt JPEG isolated; exit=2; succeeded=2; failed=1\n'
    printf '[pass] AArch64 batch evidence is functional QEMU user-mode evidence only\n'
    printf '[pass] no QEMU throughput, latency, RSS, or native-board conclusion published\n'
    printf '[pass] fresh run root=%s\n' "${run_root}"
  } 2>&1 | tee "${evidence}"
}

action="${1:-help}"
case "${action}" in
  help|-h|--help)
    usage
    ;;
  doctor)
    doctor
    ;;
  build)
    doctor
    configure_and_build
    ;;
  clean-build)
    doctor
    clean_build
    ;;
  inspect)
    doctor
    inspect_artifacts
    ;;
  smoke)
    doctor
    run_smokes
    ;;
  infer)
    doctor
    run_full_inference
    ;;
  batch)
    doctor
    run_batch_acceptance
    ;;
  all)
    doctor
    clean_build
    inspect_artifacts
    run_smokes
    run_full_inference
    run_batch_acceptance
    ;;
  *)
    fail "action" \
      "help, doctor, build, clean-build, inspect, smoke, infer, batch, or all" \
      "${action}" "run stage2_aarch64.sh help"
    ;;
esac
