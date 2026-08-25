foreach(required_variable CLI CONFIG IMAGE OUTPUT_ROOT)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_profile.cmake requires -D${required_variable}=...")
  endif()
endforeach()

if("${OUTPUT_ROOT}" STREQUAL "")
  message(FATAL_ERROR "OUTPUT_ROOT must not be empty")
endif()

get_filename_component(output_root "${OUTPUT_ROOT}" ABSOLUTE)
if("${output_root}" STREQUAL "/" OR
   "${output_root}" MATCHES "^[A-Za-z]:[/\\]?$" )
  message(FATAL_ERROR
    "Refusing to use filesystem root as OUTPUT_ROOT: '${output_root}'")
endif()

string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef run_id)
set(output_dir "${output_root}/s2_01_${run_id}/nested profile output")
set(profile_prefix "${output_dir}/yolov8_cpu_profile")

execute_process(
  COMMAND "${CLI}"
    --config "${CONFIG}"
    --image "${IMAGE}"
    --profile
    --profile-prefix "${profile_prefix}"
    --profile-runs 1
  RESULT_VARIABLE profile_result
  OUTPUT_VARIABLE profile_stdout
  ERROR_VARIABLE profile_stderr
)
if(NOT "${profile_result}" MATCHES "^-?[0-9]+$" OR
   NOT profile_result EQUAL 0)
  message(FATAL_ERROR
    "S2-01 C++ profile smoke failed. result=${profile_result}\n"
    "stdout:\n${profile_stdout}\n"
    "stderr:\n${profile_stderr}")
endif()

string(LENGTH "${profile_stdout}" stdout_length)
if(stdout_length GREATER 8192)
  message(FATAL_ERROR
    "Profile CLI summary is unexpectedly large (${stdout_length} bytes). "
    "Do not print the ORT trace document.")
endif()

string(REGEX MATCH
  "profile_trace_path: ([^\r\n]+)" profile_path_match
  "${profile_stdout}")
if("${profile_path_match}" STREQUAL "")
  message(FATAL_ERROR
    "Profile CLI did not report profile_trace_path.\n"
    "stdout:\n${profile_stdout}\n"
    "stderr:\n${profile_stderr}")
endif()
set(profile_path "${CMAKE_MATCH_1}")

if(NOT EXISTS "${profile_path}" OR IS_DIRECTORY "${profile_path}")
  message(FATAL_ERROR
    "profile_trace_path: expected an existing regular file, actual "
    "'${profile_path}'.")
endif()
file(SIZE "${profile_path}" profile_size)
if(profile_size EQUAL 0)
  message(FATAL_ERROR
    "profile_trace_path: expected non-empty ORT JSON, actual 0 bytes at "
    "'${profile_path}'.")
endif()

foreach(required_text
    "S2-01 ORT profiling completed"
    "profile_runs: 1"
    "model_id: yolov8n_neu_det_final_train_2"
    "declared_model_sha256: 7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68"
    "actual_provider: CPUExecutionProvider"
    "profiling_overhead: enabled")
  string(FIND "${profile_stdout}" "${required_text}" text_position)
  if(text_position EQUAL -1)
    message(FATAL_ERROR
      "Profile CLI summary is missing '${required_text}'.\n"
      "stdout:\n${profile_stdout}")
  endif()
endforeach()

message(STATUS
  "S2-01 C++ profile smoke passed: actual ORT trace='${profile_path}', "
  "size=${profile_size} bytes, one finite-output product run.")
