foreach(required_variable
    CLI CONFIG IMAGE OUTPUT_ROOT PYTHON JSON_VALIDATOR IMAGE_PROBE)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_single_image_outputs.cmake requires "
      "-D${required_variable}=...")
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
set(nested_output_dir
  "${output_root}/s1_05_${run_id}/nested outputs")
set(json_path "${nested_output_dir}/crazing_241.json")
set(image_path "${nested_output_dir}/crazing_241.png")

function(assert_success_output stdout_text overwrite_value)
  set(required_texts
    "S1-05 single-image detection completed"
    "actual_provider: CPUExecutionProvider"
    "detection_count:"
    "output_json:"
    "output_image:"
    "overwrite_existing: ${overwrite_value}"
    "scope:"
  )
  foreach(required_text IN LISTS required_texts)
    string(FIND "${stdout_text}" "${required_text}" position)
    if(position EQUAL -1)
      message(FATAL_ERROR
        "S1-05 CLI output is missing '${required_text}'.\n"
        "stdout:\n${stdout_text}")
    endif()
  endforeach()

  string(LENGTH "${stdout_text}" stdout_length)
  if(stdout_length GREATER 8192)
    message(FATAL_ERROR
      "S1-05 CLI summary is unexpectedly large (${stdout_length} bytes). "
      "Do not print full tensors or full JSON documents.")
  endif()
endfunction()

function(assert_regular_nonempty_file path object_name)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
    message(FATAL_ERROR
      "${object_name}: expected a regular output file at '${path}'.")
  endif()
  file(SIZE "${path}" file_size)
  if(file_size EQUAL 0)
    message(FATAL_ERROR
      "${object_name}: expected a non-empty output file at '${path}'.")
  endif()
endfunction()

function(run_json_and_image_validators)
  execute_process(
    COMMAND "${PYTHON}" -m json.tool "${json_path}"
    RESULT_VARIABLE json_tool_result
    OUTPUT_VARIABLE json_tool_stdout
    ERROR_VARIABLE json_tool_stderr
  )
  if(NOT "${json_tool_result}" MATCHES "^-?[0-9]+$" OR
     NOT json_tool_result EQUAL 0)
    message(FATAL_ERROR
      "python -m json.tool rejected '${json_path}'. "
      "result=${json_tool_result}\n"
      "stdout:\n${json_tool_stdout}\n"
      "stderr:\n${json_tool_stderr}")
  endif()

  execute_process(
    COMMAND "${PYTHON}" "${JSON_VALIDATOR}" "${json_path}"
      --expected-image "${IMAGE}"
    RESULT_VARIABLE validator_result
    OUTPUT_VARIABLE validator_stdout
    ERROR_VARIABLE validator_stderr
  )
  if(NOT "${validator_result}" MATCHES "^-?[0-9]+$" OR
     NOT validator_result EQUAL 0)
    message(FATAL_ERROR
      "Frozen detection JSON validation failed. "
      "result=${validator_result}\n"
      "stdout:\n${validator_stdout}\n"
      "stderr:\n${validator_stderr}")
  endif()

  execute_process(
    COMMAND "${IMAGE_PROBE}" "${image_path}"
    RESULT_VARIABLE image_probe_result
    OUTPUT_VARIABLE image_probe_stdout
    ERROR_VARIABLE image_probe_stderr
  )
  if(NOT "${image_probe_result}" MATCHES "^-?[0-9]+$" OR
     NOT image_probe_result EQUAL 0)
    message(FATAL_ERROR
      "OpenCV could not validate the generated visualization. "
      "result=${image_probe_result}\n"
      "stdout:\n${image_probe_stdout}\n"
      "stderr:\n${image_probe_stderr}")
  endif()
endfunction()

set(base_arguments
  --config "${CONFIG}"
  --image "${IMAGE}"
  --output-json "${json_path}"
  --output-image "${image_path}"
)

execute_process(
  COMMAND "${CLI}" ${base_arguments}
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first_stdout
  ERROR_VARIABLE first_stderr
)
if(NOT "${first_result}" MATCHES "^-?[0-9]+$" OR
   NOT first_result EQUAL 0)
  message(FATAL_ERROR
    "First S1-05 single-image run failed. result=${first_result}\n"
    "stdout:\n${first_stdout}\n"
    "stderr:\n${first_stderr}")
endif()
assert_success_output("${first_stdout}" "false")
assert_regular_nonempty_file("${json_path}" "detection_json")
assert_regular_nonempty_file("${image_path}" "visualization")
run_json_and_image_validators()

file(SHA256 "${json_path}" first_json_sha256)
file(SHA256 "${image_path}" first_image_sha256)

execute_process(
  COMMAND "${CLI}" ${base_arguments}
  RESULT_VARIABLE refusal_result
  OUTPUT_VARIABLE refusal_stdout
  ERROR_VARIABLE refusal_stderr
)
if(NOT "${refusal_result}" MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR
    "Existing-output refusal did not start correctly. "
    "result=${refusal_result}\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()
if(refusal_result EQUAL 0)
  message(FATAL_ERROR
    "Second run without --overwrite unexpectedly succeeded.\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()
set(refusal_output "${refusal_stdout}\n${refusal_stderr}")
string(FIND "${refusal_output}" "already exists" refusal_position)
if(refusal_position EQUAL -1)
  message(FATAL_ERROR
    "Existing-output refusal did not contain 'already exists'.\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()

file(SHA256 "${json_path}" refused_json_sha256)
file(SHA256 "${image_path}" refused_image_sha256)
if(NOT "${refused_json_sha256}" STREQUAL "${first_json_sha256}" OR
   NOT "${refused_image_sha256}" STREQUAL "${first_image_sha256}")
  message(FATAL_ERROR
    "The refused run modified an existing output. Expected both SHA-256 "
    "values to remain unchanged.")
endif()

file(WRITE "${json_path}" "S1-05 overwrite sentinel: JSON\n")
file(WRITE "${image_path}" "S1-05 overwrite sentinel: image\n")

execute_process(
  COMMAND "${CLI}" ${base_arguments} --overwrite
  RESULT_VARIABLE overwrite_result
  OUTPUT_VARIABLE overwrite_stdout
  ERROR_VARIABLE overwrite_stderr
)
if(NOT "${overwrite_result}" MATCHES "^-?[0-9]+$" OR
   NOT overwrite_result EQUAL 0)
  message(FATAL_ERROR
    "S1-05 --overwrite run failed. result=${overwrite_result}\n"
    "stdout:\n${overwrite_stdout}\n"
    "stderr:\n${overwrite_stderr}")
endif()
assert_success_output("${overwrite_stdout}" "true")
assert_regular_nonempty_file("${json_path}" "overwritten_detection_json")
assert_regular_nonempty_file("${image_path}" "overwritten_visualization")
run_json_and_image_validators()

file(SHA256 "${json_path}" overwrite_json_sha256)
file(SHA256 "${image_path}" overwrite_image_sha256)
if(NOT "${overwrite_json_sha256}" STREQUAL "${first_json_sha256}")
  message(FATAL_ERROR
    "Repeated successful run produced different JSON bytes. "
    "first=${first_json_sha256}, overwrite=${overwrite_json_sha256}")
endif()
if(NOT "${overwrite_image_sha256}" STREQUAL "${first_image_sha256}")
  message(FATAL_ERROR
    "Repeated successful run produced different visualization bytes. "
    "first=${first_image_sha256}, overwrite=${overwrite_image_sha256}")
endif()

message(STATUS
  "S1-05 fixed single-image CLI passed: nested directory creation, "
  "default overwrite refusal, explicit deterministic overwrite, Python "
  "JSON parsing/schema validation, and OpenCV image readability.")
