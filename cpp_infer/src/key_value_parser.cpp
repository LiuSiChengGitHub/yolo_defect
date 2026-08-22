#include "key_value_parser.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace yolo_defect_cpp {
namespace detail {
namespace {

std::string join_known_fields(const std::set<std::string>& fields) {
  std::ostringstream output;
  bool first = true;
  for (const std::string& field : fields) {
    if (!first) {
      output << ", ";
    }
    output << field;
    first = false;
  }
  return output.str();
}

std::string strip_comment(const std::string& line) {
  const std::string clean_line = trim(line);
  if (!clean_line.empty() && clean_line.front() == '#') {
    return "";
  }
  return line;
}

std::string error_prefix(const std::string& schema_name,
                         const std::filesystem::path& declaration_path,
                         int line_number,
                         const std::string& field_name) {
  std::ostringstream output;
  output << schema_name << " '" << declaration_path.string() << "'";
  if (line_number > 0) {
    output << ", line " << line_number;
  }
  if (!field_name.empty()) {
    output << ", field '" << field_name << "'";
  }
  output << ": ";
  return output.str();
}

[[noreturn]] void throw_parse_error(
    const std::string& schema_name,
    const std::filesystem::path& declaration_path,
    int line_number,
    const std::string& field_name,
    const std::string& problem,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  throw std::runtime_error(
      error_prefix(schema_name, declaration_path, line_number, field_name) +
      problem + "; expected " + expected + "; actual '" + actual +
      "'; action: " + action + ".");
}

}  // namespace

std::string trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t start = value.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return "";
  }
  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(start, end - start + 1);
}

ParsedKeyValueFile parse_key_value_file(
    const std::filesystem::path& declaration_path,
    const std::string& schema_name,
    const std::set<std::string>& known_fields) {
  ParsedKeyValueFile parsed;
  parsed.declaration_path =
      std::filesystem::absolute(declaration_path).lexically_normal();

  std::ifstream input(parsed.declaration_path);
  if (!input) {
    throw_parse_error(schema_name, parsed.declaration_path, 0, "",
                      "cannot open declaration file", "a readable file",
                      "not readable or missing",
                      "check the supplied path and file permissions");
  }

  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line_number == 1 && line.size() >= 3 &&
        static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF) {
      line.erase(0, 3);
    }
    const std::string clean_line = trim(strip_comment(line));
    if (clean_line.empty()) {
      continue;
    }

    const std::size_t separator = clean_line.find('=');
    if (separator == std::string::npos) {
      throw_parse_error(schema_name, parsed.declaration_path, line_number, "",
                        "malformed declaration", "key = value", clean_line,
                        "add exactly one key/value separator");
    }

    const std::string key = trim(clean_line.substr(0, separator));
    const std::string value = trim(clean_line.substr(separator + 1));
    if (key.empty() || value.empty()) {
      throw_parse_error(schema_name, parsed.declaration_path, line_number, key,
                        "empty key or value", "a non-empty key and value",
                        clean_line, "fill in both sides of '='");
    }

    if (known_fields.find(key) == known_fields.end()) {
      throw_parse_error(
          schema_name, parsed.declaration_path, line_number, key,
          "unknown field", "one of [" + join_known_fields(known_fields) + "]",
          key, "remove it or replace it with a supported field name");
    }

    const auto existing = parsed.fields.find(key);
    if (existing != parsed.fields.end()) {
      throw_parse_error(
          schema_name, parsed.declaration_path, line_number, key,
          "duplicate field", "exactly one declaration",
          "also declared on line " +
              std::to_string(existing->second.line_number),
          "keep only one declaration for this field");
    }

    parsed.fields.emplace(key, ParsedField{value, line_number});
  }

  return parsed;
}

const ParsedField& require_field(const ParsedKeyValueFile& parsed,
                                 const std::string& schema_name,
                                 const std::string& field_name) {
  const auto found = parsed.fields.find(field_name);
  if (found == parsed.fields.end()) {
    throw_parse_error(schema_name, parsed.declaration_path, 0, field_name,
                      "missing required field", field_name + " = <value>",
                      "<missing>", "add the required declaration");
  }
  return found->second;
}

int parse_integer_field(const ParsedKeyValueFile& parsed,
                        const std::string& schema_name,
                        const std::string& field_name) {
  const ParsedField& field = require_field(parsed, schema_name, field_name);
  std::size_t parsed_chars = 0;
  int result = 0;
  try {
    result = std::stoi(field.value, &parsed_chars);
  } catch (const std::exception&) {
    throw_field_error(parsed, schema_name, field_name, "invalid integer",
                      "a base-10 integer", field.value,
                      "replace it with a valid integer");
  }
  if (parsed_chars != field.value.size()) {
    throw_field_error(parsed, schema_name, field_name, "invalid integer",
                      "a base-10 integer", field.value,
                      "remove non-numeric suffixes");
  }
  return result;
}

double parse_number_field(const ParsedKeyValueFile& parsed,
                          const std::string& schema_name,
                          const std::string& field_name) {
  const ParsedField& field = require_field(parsed, schema_name, field_name);
  std::size_t parsed_chars = 0;
  double result = 0.0;
  try {
    result = std::stod(field.value, &parsed_chars);
  } catch (const std::exception&) {
    throw_field_error(parsed, schema_name, field_name, "invalid number",
                      "a finite number", field.value,
                      "replace it with a finite decimal value");
  }
  if (parsed_chars != field.value.size() || !std::isfinite(result)) {
    throw_field_error(parsed, schema_name, field_name, "invalid number",
                      "a finite number", field.value,
                      "remove non-numeric text, NaN, or Infinity");
  }
  return result;
}

std::vector<std::int64_t> parse_shape_field(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name) {
  const ParsedField& field = require_field(parsed, schema_name, field_name);
  if (!field.value.empty() &&
      (field.value.front() == ',' || field.value.back() == ',')) {
    throw_field_error(parsed, schema_name, field_name,
                      "empty shape dimension",
                      "comma-separated positive integers", field.value,
                      "remove the extra comma or supply the missing dimension");
  }
  std::vector<std::int64_t> shape;
  std::stringstream stream(field.value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string dimension = trim(item);
    if (dimension.empty()) {
      throw_field_error(parsed, schema_name, field_name,
                        "empty shape dimension",
                        "comma-separated positive integers", field.value,
                        "fill in every shape dimension");
    }

    std::size_t parsed_chars = 0;
    long long value = 0;
    try {
      value = std::stoll(dimension, &parsed_chars);
    } catch (const std::exception&) {
      throw_field_error(parsed, schema_name, field_name,
                        "invalid shape dimension",
                        "comma-separated positive integers", field.value,
                        "replace every dimension with a positive integer");
    }
    if (parsed_chars != dimension.size() || value <= 0) {
      throw_field_error(parsed, schema_name, field_name,
                        "invalid shape dimension",
                        "comma-separated positive integers", field.value,
                        "use static dimensions greater than zero");
    }
    shape.push_back(static_cast<std::int64_t>(value));
  }

  if (shape.empty()) {
    throw_field_error(parsed, schema_name, field_name, "empty shape",
                      "at least one positive dimension", field.value,
                      "declare the complete tensor shape");
  }
  return shape;
}

std::vector<std::string> parse_list_field(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name) {
  const ParsedField& field = require_field(parsed, schema_name, field_name);
  if (!field.value.empty() &&
      (field.value.front() == ',' || field.value.back() == ',')) {
    throw_field_error(parsed, schema_name, field_name, "empty list item",
                      "comma-separated non-empty values", field.value,
                      "remove the extra comma or supply the missing value");
  }
  std::vector<std::string> values;
  std::stringstream stream(field.value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string value = trim(item);
    if (value.empty()) {
      throw_field_error(parsed, schema_name, field_name, "empty list item",
                        "comma-separated non-empty values", field.value,
                        "remove the extra comma or supply the missing value");
    }
    values.push_back(value);
  }
  if (values.empty()) {
    throw_field_error(parsed, schema_name, field_name, "empty list",
                      "at least one value", field.value,
                      "declare at least one value");
  }
  return values;
}

std::filesystem::path resolve_declared_path(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name) {
  const ParsedField& field = require_field(parsed, schema_name, field_name);
  std::filesystem::path result(field.value);
  if (result.is_relative()) {
    result = parsed.declaration_path.parent_path() / result;
  }
  return std::filesystem::absolute(result).lexically_normal();
}

[[noreturn]] void throw_field_error(
    const ParsedKeyValueFile& parsed,
    const std::string& schema_name,
    const std::string& field_name,
    const std::string& problem,
    const std::string& expected,
    const std::string& actual,
    const std::string& action) {
  int line_number = 0;
  const auto found = parsed.fields.find(field_name);
  if (found != parsed.fields.end()) {
    line_number = found->second.line_number;
  }
  throw_parse_error(schema_name, parsed.declaration_path, line_number,
                    field_name, problem, expected, actual, action);
}

}  // namespace detail
}  // namespace yolo_defect_cpp
