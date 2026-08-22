foreach(required_variable CLI ARGS EXPECTED_TEXT)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_cli_arguments_failure.cmake requires "
      "-D${required_variable}=...")
  endif()
endforeach()

if("${CLI}" STREQUAL "")
  message(FATAL_ERROR "CLI must not be empty")
endif()
if("${ARGS}" STREQUAL "")
  message(FATAL_ERROR "ARGS must contain at least one CLI argument")
endif()
if("${EXPECTED_TEXT}" STREQUAL "")
  message(FATAL_ERROR "EXPECTED_TEXT must not be empty")
endif()

string(REPLACE "|" ";" cli_arguments "${ARGS}")
execute_process(
  COMMAND "${CLI}" ${cli_arguments}
  RESULT_VARIABLE child_result
  OUTPUT_VARIABLE child_stdout
  ERROR_VARIABLE child_stderr
)

if(NOT "${child_result}" MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR
    "CLI did not start correctly. result='${child_result}'\n"
    "arguments='${ARGS}'\n"
    "stdout:\n${child_stdout}\n"
    "stderr:\n${child_stderr}")
endif()

if(child_result EQUAL 0)
  message(FATAL_ERROR
    "Expected CLI failure but it exited 0.\n"
    "arguments='${ARGS}'\n"
    "stdout:\n${child_stdout}\n"
    "stderr:\n${child_stderr}")
endif()

set(combined_output "${child_stdout}\n${child_stderr}")
string(FIND "${combined_output}" "${EXPECTED_TEXT}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "CLI failed, but output did not contain '${EXPECTED_TEXT}'.\n"
    "exit=${child_result}\n"
    "arguments='${ARGS}'\n"
    "stdout:\n${child_stdout}\n"
    "stderr:\n${child_stderr}")
endif()

message(STATUS
  "Observed expected nonzero exit ${child_result} and text "
  "'${EXPECTED_TEXT}' for arguments '${ARGS}'.")
