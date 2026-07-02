#include "yolo_defect_cpp/config_loader.h"

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace yolo_defect_cpp {
namespace {

const std::set<std::string>& known_keys() {
  static const std::set<std::string> keys = {
      "input_width", "input_height", "class_names",
      "score_threshold", "nms_threshold", "backend"};
  return keys;
}

std::string trim(const std::string& value) {
  const std::string whitespace = " \t\r\n";
  const std::size_t start = value.find_first_not_of(whitespace);
  if (start == std::string::npos) {
    return "";
  }

  const std::size_t end = value.find_last_not_of(whitespace);
  return value.substr(start, end - start + 1);
}

std::string strip_comment(const std::string& line) {
  const std::size_t comment_start = line.find('#');
  if (comment_start == std::string::npos) {
    return line;
  }
  return line.substr(0, comment_start);
}

std::string line_prefix(int line_number) {
  return "Config line " + std::to_string(line_number) + ": ";
}

int parse_int(const std::string& key, const std::string& value) {
  std::size_t parsed_chars = 0;
  int parsed_value = 0;
  try {
    parsed_value = std::stoi(value, &parsed_chars);
  } catch (const std::exception&) {
    throw std::runtime_error("Config key '" + key + "' must be an integer.");
  }

  if (parsed_chars != value.size()) {
    throw std::runtime_error("Config key '" + key + "' must be an integer.");
  }
  return parsed_value;
}

double parse_double(const std::string& key, const std::string& value) {
  std::size_t parsed_chars = 0;
  double parsed_value = 0.0;
  try {
    parsed_value = std::stod(value, &parsed_chars);
  } catch (const std::exception&) {
    throw std::runtime_error("Config key '" + key + "' must be a number.");
  }

  if (parsed_chars != value.size()) {
    throw std::runtime_error("Config key '" + key + "' must be a number.");
  }
  return parsed_value;
}

std::vector<std::string> parse_class_names(const std::string& value) {
  std::vector<std::string> names;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const std::string name = trim(item);
    if (name.empty()) {
      throw std::runtime_error("Config key 'class_names' contains an empty class name.");
    }
    names.push_back(name);
  }

  if (names.empty()) {
    throw std::runtime_error("Config key 'class_names' must contain at least one class.");
  }
  return names;
}

void require_key(const std::map<std::string, std::string>& values,
                 const std::string& key) {
  if (values.find(key) == values.end()) {
    throw std::runtime_error("Missing required config key: " + key);
  }
}

void validate_positive_dimension(const std::string& key, int value) {
  if (value <= 0) {
    throw std::runtime_error("Config key '" + key + "' must be greater than 0.");
  }
}

void validate_threshold(const std::string& key, double value) {
  if (value < 0.0 || value > 1.0) {
    throw std::runtime_error("Config key '" + key + "' must be in [0, 1].");
  }
}

}  // namespace

RuntimeConfig load_config(const std::string& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("Failed to open config file: " + config_path);
  }

  std::map<std::string, std::string> values;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::string clean_line = trim(strip_comment(line));
    if (clean_line.empty()) {
      continue;
    }

    const std::size_t separator = clean_line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error(line_prefix(line_number) +
                               "expected 'key = value'.");
    }

    const std::string key = trim(clean_line.substr(0, separator));
    const std::string value = trim(clean_line.substr(separator + 1));
    if (key.empty() || value.empty()) {
      throw std::runtime_error(line_prefix(line_number) +
                               "expected non-empty key and value.");
    }

    if (known_keys().find(key) == known_keys().end()) {
      throw std::runtime_error(line_prefix(line_number) +
                               "unknown config key: " + key);
    }

    if (values.find(key) != values.end()) {
      throw std::runtime_error(line_prefix(line_number) +
                               "duplicate config key: " + key);
    }
    values[key] = value;
  }

  require_key(values, "input_width");
  require_key(values, "input_height");
  require_key(values, "class_names");
  require_key(values, "score_threshold");
  require_key(values, "nms_threshold");
  require_key(values, "backend");

  RuntimeConfig config;
  config.input_width = parse_int("input_width", values.at("input_width"));
  config.input_height = parse_int("input_height", values.at("input_height"));
  config.class_names = parse_class_names(values.at("class_names"));
  config.score_threshold = parse_double("score_threshold", values.at("score_threshold"));
  config.nms_threshold = parse_double("nms_threshold", values.at("nms_threshold"));
  config.backend = values.at("backend");

  validate_positive_dimension("input_width", config.input_width);
  validate_positive_dimension("input_height", config.input_height);
  validate_threshold("score_threshold", config.score_threshold);
  validate_threshold("nms_threshold", config.nms_threshold);
  if (trim(config.backend).empty()) {
    throw std::runtime_error("Config key 'backend' must be non-empty.");
  }

  return config;
}

}  // namespace yolo_defect_cpp
