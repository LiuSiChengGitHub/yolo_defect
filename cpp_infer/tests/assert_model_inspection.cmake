foreach(required_variable CLI CONFIG)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR
      "assert_model_inspection.cmake requires -D${required_variable}=...")
  endif()
endforeach()

execute_process(
  COMMAND "${CLI}" --config "${CONFIG}" --inspect-model
  RESULT_VARIABLE child_result
  OUTPUT_VARIABLE child_stdout
  ERROR_VARIABLE child_stderr
)

if(NOT "${child_result}" MATCHES "^-?[0-9]+$" OR
   NOT child_result EQUAL 0)
  message(FATAL_ERROR
    "Model inspection did not complete successfully. result=${child_result}\n"
    "stdout:\n${child_stdout}\n"
    "stderr:\n${child_stderr}")
endif()

set(required_texts
  "C++ runtime model inspection"
  "ort_version: 1.19.2"
  "configured_provider: cpu"
  "CPUExecutionProvider"
  "session_provider: CPUExecutionProvider"
  "input_count: 1"
  "input[0].name: images"
  "input[0].shape: [1,3,800,800]"
  "input[0].dtype: float32"
  "output_count: 1"
  "output[0].name: output0"
  "output[0].shape: [1,10,13125]"
  "output[0].dtype: float32"
  "metadata_contract_validation: passed"
  "no input tensor, backend run, inference result, or postprocess"
)

foreach(required_text IN LISTS required_texts)
  string(FIND "${child_stdout}" "${required_text}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "Model inspection output is missing '${required_text}'.\n"
      "stdout:\n${child_stdout}\n"
      "stderr:\n${child_stderr}")
  endif()
endforeach()

message(STATUS
  "Observed real ORT 1.19.2 session metadata and a passed contract.")
