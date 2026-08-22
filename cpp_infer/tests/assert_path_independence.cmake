foreach(required_variable CLI CONFIG FIRST_CWD SECOND_CWD)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_path_independence.cmake requires -D${required_variable}=...")
  endif()
endforeach()

execute_process(
  COMMAND "${CLI}" --config "${CONFIG}"
  WORKING_DIRECTORY "${FIRST_CWD}"
  RESULT_VARIABLE first_result
  OUTPUT_VARIABLE first_stdout
  ERROR_VARIABLE first_stderr
)

execute_process(
  COMMAND "${CLI}" --config "${CONFIG}"
  WORKING_DIRECTORY "${SECOND_CWD}"
  RESULT_VARIABLE second_result
  OUTPUT_VARIABLE second_stdout
  ERROR_VARIABLE second_stderr
)

foreach(result_variable first_result second_result)
  if(NOT "${${result_variable}}" MATCHES "^-?[0-9]+$" OR
     NOT ${result_variable} EQUAL 0)
    message(FATAL_ERROR
      "CLI path-independence probe failed to run successfully.\n"
      "first result=${first_result}\n${first_stdout}\n${first_stderr}\n"
      "second result=${second_result}\n${second_stdout}\n${second_stderr}")
  endif()
endforeach()

string(REGEX MATCH "artifact_spec_path: [^\r\n]*"
  first_artifact_path "${first_stdout}")
string(REGEX MATCH "artifact_spec_path: [^\r\n]*"
  second_artifact_path "${second_stdout}")
string(REGEX MATCH "model_path: [^\r\n]*"
  first_model_path "${first_stdout}")
string(REGEX MATCH "model_path: [^\r\n]*"
  second_model_path "${second_stdout}")

if(first_artifact_path STREQUAL "" OR first_model_path STREQUAL "")
  message(FATAL_ERROR
    "CLI summary did not expose resolved artifact/model paths.\n${first_stdout}")
endif()

if(NOT first_artifact_path STREQUAL second_artifact_path OR
   NOT first_model_path STREQUAL second_model_path)
  message(FATAL_ERROR
    "Resolved paths changed with working directory.\n"
    "first artifact: ${first_artifact_path}\n"
    "second artifact: ${second_artifact_path}\n"
    "first model: ${first_model_path}\n"
    "second model: ${second_model_path}")
endif()

message(STATUS
  "Resolved artifact and model paths are identical across working directories.")
