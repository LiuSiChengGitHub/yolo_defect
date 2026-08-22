foreach(required_variable CLI CONFIG IMAGE)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_raw_output_summary.cmake requires "
      "-D${required_variable}=...")
  endif()
endforeach()

execute_process(
  COMMAND "${CLI}"
    --config "${CONFIG}"
    --image "${IMAGE}"
    --raw-output-summary
  RESULT_VARIABLE child_result
  OUTPUT_VARIABLE child_stdout
  ERROR_VARIABLE child_stderr
)

if(NOT "${child_result}" MATCHES "^-?[0-9]+$" OR
   NOT child_result EQUAL 0)
  message(FATAL_ERROR
    "Raw inference did not complete successfully. result=${child_result}\n"
    "stdout:\n${child_stdout}\n"
    "stderr:\n${child_stderr}")
endif()

set(required_texts
  "S1-03 raw output summary"
  "input_shape: [1,3,800,800]"
  "input_elements: 1920000"
  "input_finite_values: 1920000/1920000"
  "input_min:"
  "input_max:"
  "output_shape: [1,10,13125]"
  "output_elements: 131250"
  "output_finite_values: 131250/131250"
  "output_min:"
  "output_max:"
  "session_run: completed"
  "raw_output_ownership: copied_to_InferenceOutput"
  "no decode, NMS, JSON, visualization, or benchmark"
)

foreach(required_text IN LISTS required_texts)
  string(FIND "${child_stdout}" "${required_text}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "Raw output summary is missing '${required_text}'.\n"
      "stdout:\n${child_stdout}\n"
      "stderr:\n${child_stderr}")
  endif()
endforeach()

string(LENGTH "${child_stdout}" stdout_length)
if(stdout_length GREATER 4096)
  message(FATAL_ERROR
    "Raw output summary is unexpectedly large (${stdout_length} bytes). "
    "Do not print the full tensor.")
endif()

message(STATUS
  "Observed one bounded, finite, independently owned raw output summary.")
