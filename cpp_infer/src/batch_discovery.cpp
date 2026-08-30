#include "yolo_defect_cpp/batch_runner.h"

#include "batch_path_safety.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

constexpr std::uintmax_t kMaximumManifestBytes = 16U * 1024U * 1024U;

[[noreturn]] void throw_discovery_error(
    const std::string& object, const std::string& expected,
    const std::string& actual, const std::string& action) {
  std::ostringstream message;
  message << "Batch discovery failed: object " << object
          << "; expected " << expected
          << "; actual " << actual
          << "; action: " << action;
  throw std::runtime_error(message.str());
}

std::string display_path(const std::filesystem::path& path) {
  try {
    return path.generic_u8string();
  } catch (const std::exception&) {
    return "<path cannot be converted to UTF-8>";
  }
}

void validate_utf8(const std::string& value, const std::string& object) {
  const auto is_continuation = [](unsigned char value_byte) {
    return value_byte >= 0x80U && value_byte <= 0xBFU;
  };
  std::size_t index = 0;
  while (index < value.size()) {
    const unsigned char first = static_cast<unsigned char>(value[index]);
    std::size_t length = 0;
    if (first <= 0x7FU) {
      length = 1;
    } else if (first >= 0xC2U && first <= 0xDFU) {
      length = 2;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      length = 3;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      length = 4;
    } else {
      throw_discovery_error(
          object, "valid UTF-8", "invalid leading byte at byte " +
              std::to_string(index),
          "save the manifest and filenames as UTF-8");
    }
    if (index + length > value.size()) {
      throw_discovery_error(
          object, "valid UTF-8", "truncated sequence at byte " +
              std::to_string(index),
          "save the complete manifest as UTF-8");
    }
    for (std::size_t offset = 1; offset < length; ++offset) {
      if (!is_continuation(
              static_cast<unsigned char>(value[index + offset]))) {
        throw_discovery_error(
            object, "valid UTF-8", "invalid continuation byte at byte " +
                std::to_string(index + offset),
            "save the manifest and filenames as UTF-8");
      }
    }
    if (length == 3) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xE0U && second < 0xA0U) ||
          (first == 0xEDU && second > 0x9FU)) {
        throw_discovery_error(
            object, "canonical UTF-8", "invalid three-byte sequence at byte " +
                std::to_string(index),
            "remove overlong or surrogate encodings");
      }
    } else if (length == 4) {
      const unsigned char second =
          static_cast<unsigned char>(value[index + 1]);
      if ((first == 0xF0U && second < 0x90U) ||
          (first == 0xF4U && second > 0x8FU)) {
        throw_discovery_error(
            object, "UTF-8 in the Unicode range",
            "invalid four-byte sequence at byte " +
                std::to_string(index),
            "remove overlong or out-of-range encodings");
      }
    }
    index += length;
  }
}

std::filesystem::path absolute_normalized(
    const std::filesystem::path& path, const std::string& object) {
  if (path.empty()) {
    throw_discovery_error(
        object, "a non-empty filesystem path", "empty",
        "provide --input-dir <dir> or --manifest <file>");
  }
  std::error_code error;
  std::filesystem::path result = std::filesystem::absolute(path, error);
  if (error) {
    throw_discovery_error(
        object, "a path resolvable from the working directory",
        display_path(path) + " (" + error.message() + ")",
        "correct the path or working directory");
  }
  return result.lexically_normal();
}

std::filesystem::path canonical_existing(
    const std::filesystem::path& path, const std::string& object) {
  const std::filesystem::path absolute = absolute_normalized(path, object);
  std::error_code error;
  const std::filesystem::path result =
      std::filesystem::canonical(absolute, error);
  if (error) {
    throw_discovery_error(
        object, "an existing accessible path",
        display_path(absolute) + " (" + error.message() + ")",
        "correct the path and check filesystem permissions");
  }
  return result;
}

std::string lowercase_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return value;
}

bool is_supported_image_path(const std::filesystem::path& path) {
  const std::string extension = lowercase_ascii(path.extension().u8string());
  return extension == ".bmp" || extension == ".jpeg" ||
         extension == ".jpg" || extension == ".png" ||
         extension == ".tif" || extension == ".tiff" ||
         extension == ".webp";
}

void add_unique_task(
    std::vector<BatchTask>& tasks,
    std::map<std::filesystem::path, std::string,
             internal::BatchPathLocationLess>& first_declarations,
    const std::filesystem::path& canonical_source,
    const std::filesystem::path& logical_path,
    const std::string& declaration) {
  const std::string canonical_utf8 = canonical_source.generic_u8string();
  validate_utf8(canonical_utf8, "canonical_source_path");
  const auto insertion =
      first_declarations.emplace(canonical_source, declaration);
  if (!insertion.second) {
    throw_discovery_error(
        "source_images", "each canonical image path exactly once",
        "duplicate '" + declaration + "' resolves to " +
            display_path(canonical_source) + "; first declared by " +
            insertion.first->second,
        "remove the duplicate manifest entry or duplicate source alias");
  }
  BatchTask task;
  task.source_path = canonical_source;
  task.logical_path = logical_path;
  tasks.push_back(std::move(task));
}

std::vector<BatchTask> discover_directory(
    const std::filesystem::path& declared_directory) {
  const std::filesystem::path declared_absolute = absolute_normalized(
      declared_directory, "input.directory");
  std::error_code declared_error;
  const std::filesystem::file_status declared_status =
      std::filesystem::symlink_status(declared_absolute, declared_error);
  if (declared_error &&
      declared_status.type() != std::filesystem::file_type::not_found) {
    throw_discovery_error(
        "input.directory", "an inspectable real directory path",
        display_path(declared_absolute) + " (" +
            declared_error.message() + ")",
        "check directory permissions and filesystem state");
  }
  if (std::filesystem::exists(declared_status)) {
    std::error_code reparse_error;
    const bool reparse = internal::batch_path_is_reparse_point(
        declared_absolute, reparse_error);
    if (reparse_error) {
      throw_discovery_error(
          "input.directory", "queryable filesystem indirection metadata",
          display_path(declared_absolute) + " (" +
              reparse_error.message() + ")",
          "check directory permissions and filesystem state");
    }
    if (std::filesystem::is_symlink(declared_status) || reparse) {
      throw_discovery_error(
          "input.directory",
          "a real directory path without symlink/reparse indirection",
          display_path(declared_absolute) +
              " is a symlink or reparse point",
          "pass the real input directory so deterministic discovery does "
          "not follow indirection");
    }
  }
  const std::filesystem::path root = canonical_existing(
      declared_absolute, "input.directory");
  std::error_code error;
  if (!std::filesystem::is_directory(root, error) || error) {
    throw_discovery_error(
        "input.directory", "an accessible directory",
        display_path(root) +
            (error ? " (" + error.message() + ")" : " (not a directory)"),
        "pass a directory to --input-dir rather than a file");
  }

  std::vector<BatchTask> tasks;
  std::map<std::filesystem::path, std::string,
           internal::BatchPathLocationLess> declarations;
  std::filesystem::recursive_directory_iterator iterator(root, error);
  const std::filesystem::recursive_directory_iterator end;
  if (error) {
    throw_discovery_error(
        "input.directory", "a recursively enumerable directory",
        display_path(root) + " (" + error.message() + ")",
        "check directory read and traversal permissions");
  }
  while (iterator != end) {
    const std::filesystem::directory_entry entry = *iterator;
    const std::filesystem::file_status status =
        entry.symlink_status(error);
    if (error) {
      throw_discovery_error(
          "input.directory.entry", "inspectable filesystem metadata",
          display_path(entry.path()) + " (" + error.message() + ")",
          "fix the inaccessible entry or its parent permissions");
    }

    if (std::filesystem::is_symlink(status)) {
      // recursive_directory_iterator does not follow directory symlinks by
      // default; disabling recursion makes that invariant explicit.
      if (entry.is_directory(error) && !error) {
        iterator.disable_recursion_pending();
      }
      error.clear();
    } else if (std::filesystem::is_regular_file(status) &&
               is_supported_image_path(entry.path())) {
      const std::filesystem::path relative =
          entry.path().lexically_relative(root).lexically_normal();
      const std::string ordering_key = relative.generic_u8string();
      validate_utf8(ordering_key, "input.directory.relative_path");
      const std::filesystem::path canonical =
          std::filesystem::canonical(entry.path(), error);
      if (error) {
        throw_discovery_error(
            "input.directory.image", "a canonicalizable regular file",
            display_path(entry.path()) + " (" + error.message() + ")",
            "check whether the image changed during discovery");
      }
      add_unique_task(tasks, declarations, canonical, relative,
                      ordering_key);
    }

    iterator.increment(error);
    if (error) {
      throw_discovery_error(
          "input.directory", "complete recursive traversal",
          display_path(root) + " (" + error.message() + ")",
          "fix the inaccessible entry instead of publishing a partial task "
          "set");
    }
  }

  std::sort(tasks.begin(), tasks.end(),
            [](const BatchTask& lhs, const BatchTask& rhs) {
              return lhs.logical_path.generic_u8string() <
                     rhs.logical_path.generic_u8string();
            });
  if (tasks.empty()) {
    throw_discovery_error(
        "input.directory", "at least one supported regular image file",
        "no .bmp/.jpeg/.jpg/.png/.tif/.tiff/.webp files under " +
            display_path(root),
        "add input images or choose the intended directory");
  }
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    tasks[index].sequence_index = index;
  }
  return tasks;
}

std::string read_manifest(const std::filesystem::path& manifest) {
  std::error_code error;
  const std::uintmax_t file_size =
      std::filesystem::file_size(manifest, error);
  if (error || file_size > kMaximumManifestBytes ||
      file_size > static_cast<std::uintmax_t>(
                      std::numeric_limits<std::streamsize>::max())) {
    throw_discovery_error(
        "input.manifest", "a readable UTF-8 file no larger than 16 MiB",
        display_path(manifest) +
            (error ? " (" + error.message() + ")"
                   : " (" + std::to_string(file_size) + " bytes)"),
        "use a bounded path-list manifest and check read permissions");
  }
  std::ifstream input(manifest, std::ios::binary);
  if (!input.is_open()) {
    throw_discovery_error(
        "input.manifest", "a readable regular file", display_path(manifest),
        "check manifest read permissions");
  }
  std::string document(static_cast<std::size_t>(file_size), '\0');
  if (!document.empty()) {
    input.read(document.data(), static_cast<std::streamsize>(document.size()));
  }
  if (!input || input.gcount() !=
                    static_cast<std::streamsize>(document.size())) {
    throw_discovery_error(
        "input.manifest", "the complete manifest contents",
        std::to_string(input.gcount()) + " of " +
            std::to_string(document.size()) + " bytes read",
        "check whether the manifest changed during discovery");
  }
  return document;
}

std::string trim_ascii_whitespace(const std::string& value) {
  const auto whitespace = [](unsigned char character) {
    return character == ' ' || character == '\t' || character == '\v' ||
           character == '\f';
  };
  std::size_t begin = 0;
  while (begin < value.size() &&
         whitespace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin &&
         whitespace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::vector<BatchTask> discover_manifest(
    const std::filesystem::path& declared_manifest) {
  const std::filesystem::path manifest = canonical_existing(
      declared_manifest, "input.manifest");
  std::error_code error;
  if (!std::filesystem::is_regular_file(manifest, error) || error) {
    throw_discovery_error(
        "input.manifest", "an accessible regular UTF-8 path-list file",
        display_path(manifest) +
            (error ? " (" + error.message() + ")" : " (not a regular file)"),
        "pass a manifest file rather than a directory or special file");
  }

  std::string document = read_manifest(manifest);
  if (document.size() >= 3U &&
      static_cast<unsigned char>(document[0]) == 0xEFU &&
      static_cast<unsigned char>(document[1]) == 0xBBU &&
      static_cast<unsigned char>(document[2]) == 0xBFU) {
    document.erase(0, 3);
  }
  validate_utf8(document, "input.manifest.contents");
  if (document.find('\0') != std::string::npos) {
    throw_discovery_error(
        "input.manifest.contents", "text paths without NUL bytes",
        "embedded NUL byte",
        "rewrite the manifest as a normal UTF-8 text file");
  }

  const std::filesystem::path base = manifest.parent_path();
  std::vector<BatchTask> tasks;
  std::map<std::filesystem::path, std::string,
           internal::BatchPathLocationLess> declarations;
  std::istringstream lines(document);
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(lines, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const std::string value = trim_ascii_whitespace(line);
    if (value.empty() || value.front() == '#') {
      continue;
    }

    std::filesystem::path relative;
    try {
      relative = std::filesystem::u8path(value);
    } catch (const std::exception& exception) {
      throw_discovery_error(
          "input.manifest.line[" + std::to_string(line_number) + "]",
          "a UTF-8 filesystem path", exception.what(),
          "correct the path text on this line");
    }
    if (relative.empty() || relative.is_absolute() ||
        relative.has_root_name() || relative.has_root_directory()) {
      throw_discovery_error(
          "input.manifest.line[" + std::to_string(line_number) + "]",
          "one non-empty relative image path", value,
          "make the path relative to the manifest directory");
    }
    if (!is_supported_image_path(relative)) {
      throw_discovery_error(
          "input.manifest.line[" + std::to_string(line_number) + "]",
          "an image path ending in .bmp/.jpeg/.jpg/.png/.tif/.tiff/.webp",
          value,
          "correct the extension or remove the non-image entry");
    }

    const std::filesystem::path candidate =
        (base / relative).lexically_normal();
    const std::filesystem::path canonical =
        std::filesystem::canonical(candidate, error);
    if (error) {
      throw_discovery_error(
          "input.manifest.line[" + std::to_string(line_number) + "]",
          "an existing accessible image file",
          value + " resolves from " + display_path(base) + " (" +
              error.message() + ")",
          "correct the relative path or restore the source image");
    }
    if (!std::filesystem::is_regular_file(canonical, error) || error) {
      throw_discovery_error(
          "input.manifest.line[" + std::to_string(line_number) + "]",
          "an existing regular image file",
          display_path(canonical) +
              (error ? " (" + error.message() + ")"
                     : " (not a regular file)"),
          "replace the entry with a regular image file");
    }
    add_unique_task(
        tasks, declarations, canonical, relative.lexically_normal(),
        "manifest line " + std::to_string(line_number) + " ('" + value +
            "')");
  }

  if (tasks.empty()) {
    throw_discovery_error(
        "input.manifest", "at least one image path",
        "only BOM, blank lines, or comments in " + display_path(manifest),
        "add a relative image path on its own line");
  }
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    tasks[index].sequence_index = index;
  }
  return tasks;
}

}  // namespace

std::string to_string(BatchInputKind value) {
  switch (value) {
    case BatchInputKind::kDirectory:
      return "directory";
    case BatchInputKind::kManifest:
      return "manifest";
  }
  throw_discovery_error(
      "input.kind", "directory or manifest", "unknown enum value",
      "initialize BatchRequest.input_kind explicitly");
}

std::string to_string(BatchItemStatus value) {
  switch (value) {
    case BatchItemStatus::kSucceeded:
      return "succeeded";
    case BatchItemStatus::kFailed:
      return "failed";
    case BatchItemStatus::kCancelled:
      return "cancelled";
  }
  throw_discovery_error(
      "item.status", "succeeded, failed, or cancelled",
      "unknown enum value", "initialize each BatchItemResult status");
}

std::string to_string(BatchStatus value) {
  switch (value) {
    case BatchStatus::kSucceeded:
      return "succeeded";
    case BatchStatus::kPartialFailure:
      return "partial_failure";
    case BatchStatus::kCancelled:
      return "cancelled";
    case BatchStatus::kFatal:
      return "fatal";
  }
  throw_discovery_error(
      "status", "succeeded, partial_failure, cancelled, or fatal",
      "unknown enum value", "derive status from final batch counts");
}

std::vector<BatchTask> discover_batch_tasks(
    BatchInputKind input_kind,
    const std::filesystem::path& input_path) {
  switch (input_kind) {
    case BatchInputKind::kDirectory:
      return discover_directory(input_path);
    case BatchInputKind::kManifest:
      return discover_manifest(input_path);
  }
  throw_discovery_error(
      "input.kind", "directory or manifest", "unknown enum value",
      "initialize BatchRequest.input_kind explicitly");
}

}  // namespace yolo_defect_cpp
