foreach(required_variable
    CLI CONFIG IMAGE OUTPUT_ROOT PYTHON JSON_VALIDATOR)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_benchmark.cmake requires -D${required_variable}=...")
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
set(output_dir "${output_root}/s1_08_${run_id}/nested benchmark output")
set(benchmark_json "${output_dir}/crazing_241.benchmark.json")

set(benchmark_arguments
  --config "${CONFIG}"
  --image "${IMAGE}"
  --benchmark
  --warmup 1
  --repeat 2
  --benchmark-json "${benchmark_json}"
)

execute_process(
  COMMAND "${CLI}" ${benchmark_arguments}
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first_stdout
  ERROR_VARIABLE first_stderr
)
if(NOT "${first_result}" MATCHES "^-?[0-9]+$" OR
   NOT first_result EQUAL 0)
  message(FATAL_ERROR
    "S1-08 short Release benchmark failed. result=${first_result}\n"
    "stdout:\n${first_stdout}\n"
    "stderr:\n${first_stderr}")
endif()

if(NOT EXISTS "${benchmark_json}" OR IS_DIRECTORY "${benchmark_json}")
  message(FATAL_ERROR
    "benchmark_json: expected a regular output file at "
    "'${benchmark_json}'.")
endif()
file(SIZE "${benchmark_json}" benchmark_size)
if(benchmark_size EQUAL 0)
  message(FATAL_ERROR
    "benchmark_json: expected a non-empty file at '${benchmark_json}'.")
endif()

string(LENGTH "${first_stdout}" stdout_length)
if(stdout_length GREATER 8192)
  message(FATAL_ERROR
    "S1-08 CLI summary is unexpectedly large (${stdout_length} bytes). "
    "Do not print raw timing samples or the complete JSON document.")
endif()

execute_process(
  COMMAND "${PYTHON}" -m json.tool "${benchmark_json}"
  RESULT_VARIABLE json_tool_result
  OUTPUT_VARIABLE json_tool_stdout
  ERROR_VARIABLE json_tool_stderr
)
if(NOT "${json_tool_result}" MATCHES "^-?[0-9]+$" OR
   NOT json_tool_result EQUAL 0)
  message(FATAL_ERROR
    "python -m json.tool rejected '${benchmark_json}'. "
    "result=${json_tool_result}\n"
    "stdout:\n${json_tool_stdout}\n"
    "stderr:\n${json_tool_stderr}")
endif()

execute_process(
  COMMAND "${PYTHON}" "${JSON_VALIDATOR}" "${benchmark_json}"
    --expected-image "${IMAGE}"
    --expected-warmup 1
    --expected-repeat 2
  RESULT_VARIABLE validator_result
  OUTPUT_VARIABLE validator_stdout
  ERROR_VARIABLE validator_stderr
)
if(NOT "${validator_result}" MATCHES "^-?[0-9]+$" OR
   NOT validator_result EQUAL 0)
  message(FATAL_ERROR
    "Strict S1-08 benchmark JSON validation failed. "
    "result=${validator_result}\n"
    "stdout:\n${validator_stdout}\n"
    "stderr:\n${validator_stderr}")
endif()

file(SHA256 "${benchmark_json}" first_sha256)

execute_process(
  COMMAND "${CLI}" ${benchmark_arguments}
  RESULT_VARIABLE refusal_result
  OUTPUT_VARIABLE refusal_stdout
  ERROR_VARIABLE refusal_stderr
)
if(NOT "${refusal_result}" MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR
    "Existing benchmark refusal did not start correctly. "
    "result=${refusal_result}\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()
if(refusal_result EQUAL 0)
  message(FATAL_ERROR
    "Second benchmark run without --overwrite unexpectedly succeeded.\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()

set(refusal_output "${refusal_stdout}\n${refusal_stderr}")
string(FIND "${refusal_output}" "already exists" refusal_position)
if(refusal_position EQUAL -1)
  message(FATAL_ERROR
    "Existing benchmark refusal did not contain 'already exists'.\n"
    "stdout:\n${refusal_stdout}\n"
    "stderr:\n${refusal_stderr}")
endif()

file(SHA256 "${benchmark_json}" refused_sha256)
if(NOT "${refused_sha256}" STREQUAL "${first_sha256}")
  message(FATAL_ERROR
    "The refused benchmark run modified existing evidence. "
    "expected SHA-256 ${first_sha256}, actual ${refused_sha256}.")
endif()

message(STATUS
  "S1-08 short benchmark passed: warmup=1, repeat=2, strict JSON and "
  "Release/CPU/memory/disclosure validation, plus default overwrite "
  "refusal without modifying the first evidence file. ${validator_stdout}")
