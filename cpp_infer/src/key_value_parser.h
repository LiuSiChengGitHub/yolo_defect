#ifndef YOLO_DEFECT_CPP_KEY_VALUE_PARSER_H_
#define YOLO_DEFECT_CPP_KEY_VALUE_PARSER_H_

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace yolo_defect_cpp {
namespace detail {

struct ParsedField {
  std::string value;
  int line_number = 0;
};

struct ParsedKeyValueFile {
  std::filesystem::path declaration_path;
  std::map<std::string, ParsedField> fields;
};

ParsedKeyValueFile parse_key_value_file(
    const std::filesystem::path& declaration_path,
    const std::string& schema_name,
    const std::set<std::string>& known_fields);

const ParsedField& require_field(const ParsedKeyValueFile& parsed,
                                 const std::string& schema_name,
                                 const std::string& field_name);

int parse_integer_field(const ParsedKeyValueFile& parsed,
                        const std::string& schema_name,
                        const std::string& field_name);

double parse_number_field(const ParsedKeyValueFile& parsed,
                          const std::string& schema_name,
                          const std::string& field_name);

std::vector<std::int64_t> parse_shape_field(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name);

std::vector<std::string> parse_list_field(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name);

std::filesystem::path resolve_declared_path(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name);

std::string trim(const std::string& value);

[[noreturn]] void throw_field_error(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name,
    const std::string& problem,
    const std::string& expected,
    const std::string& actual,
    const std::string& action);

}  // namespace detail
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_KEY_VALUE_PARSER_H_
