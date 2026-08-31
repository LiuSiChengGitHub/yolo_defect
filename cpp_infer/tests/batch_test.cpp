#include "yolo_defect_cpp/batch_runner.h"
#include "yolo_defect_cpp/batch_writer.h"

#include "batch_executor.h"
#include "batch_path_safety.h"
#include "bounded_queue.h"

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace yolo_defect_cpp {
namespace {

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(const std::string& label) {
    const auto suffix = std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count();
    path_ = std::filesystem::temp_directory_path() /
            ("yolo_defect_s2_03_" + label + "_" +
             std::to_string(suffix));
    if (!std::filesystem::create_directories(path_)) {
      throw std::runtime_error(
          "could not create temporary test directory: " + path_.string());
    }
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path,
                const std::string& contents = "placeholder") {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    throw std::runtime_error("could not create test file: " + path.string());
  }
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output.good()) {
    throw std::runtime_error("could not write test file: " + path.string());
  }
}

template <typename Callable>
std::string capture_runtime_error(Callable&& callable) {
  try {
    callable();
  } catch (const std::runtime_error& error) {
    return error.what();
  } catch (...) {
    ADD_FAILURE() << "Expected std::runtime_error, but another exception "
                     "type was thrown.";
    return {};
  }
  ADD_FAILURE() << "Expected std::runtime_error, but no exception was thrown.";
  return {};
}

template <typename Predicate>
bool wait_until(Predicate&& predicate) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::yield();
  }
  return predicate();
}

std::vector<std::filesystem::path> task_logical_paths(
    const std::vector<BatchTask>& tasks) {
  std::vector<std::filesystem::path> paths;
  for (const BatchTask& task : tasks) {
    paths.push_back(task.logical_path);
  }
  return paths;
}

BatchSummary make_valid_summary(const std::filesystem::path& root) {
  BatchSummary summary;
  summary.timestamp_utc = "2026-08-30T00:00:00Z";
  summary.status = BatchStatus::kSucceeded;
  summary.command_arguments = {"yolo_defect_cpp", "--batch"};

  summary.environment.hostname = "test-host";
  summary.environment.processor = "test-processor";
  summary.environment.logical_cpu_count = 8;
  summary.environment.os_name = "test-os";
  summary.environment.os_version = "1";
  summary.environment.target_architecture = "x86_64";
  summary.environment.runtime_kernel_architecture = "x86_64";
  summary.environment.execution_context = "native_or_unknown";
  summary.environment.compiler_id = "test-compiler";
  summary.environment.compiler_version = "1";
  summary.environment.build_type = "Release";
  summary.environment.cxx_standard = 17;
  summary.environment.opencv_version = "4.10.0";
  summary.environment.onnxruntime_version = "1.19.2";

  summary.runtime.config_path = root / "config.txt";
  summary.runtime.requested_provider = "cpu";
  summary.runtime.actual_provider = "CPUExecutionProvider";
  summary.runtime.provider_evidence = "test evidence";
  summary.runtime.execution_mode = "sequential";
  summary.runtime.intra_op_num_threads = 1;
  summary.runtime.inter_op_num_threads = 1;
  summary.runtime.graph_optimization_level = "all";
  summary.runtime.score_threshold = 0.25;
  summary.runtime.nms_threshold = 0.45;
  summary.runtime.nms_mode = "class_agnostic";
  summary.runtime.requested_workers = 1;
  summary.runtime.effective_workers = 1;
  summary.runtime.session_count = 1;
  summary.runtime.session_initialization_ms = {2.0};

  summary.model.model_id = "test-model";
  summary.model.model_family = "yolov8_detection";
  summary.model.model_path = root / "model.onnx";
  summary.model.declared_sha256 = std::string(64, 'A');
  summary.model.opset = 12;
  summary.model.input_name = "images";
  summary.model.input_shape = {1, 3, 800, 800};
  summary.model.input_dtype = "float32";
  summary.model.input_layout = "nchw";

  summary.input.kind = BatchInputKind::kDirectory;
  summary.input.source_path = root / "input";
  summary.input.ordering =
      "recursive UTF-8 generic relative-path lexical order; supported "
      "regular files only; symlinks not followed";
  summary.output.directory = root / "output";
  summary.output.batch_summary_path = root / "summary.json";
  summary.output.item_directory = root / "output" / "items";

  summary.counts.discovered = 1;
  summary.counts.enqueued = 1;
  summary.counts.started = 1;
  summary.counts.succeeded = 1;
  summary.queue.capacity = 1;
  summary.queue.peak_depth = 1;
  summary.queue.producer_wait_count = 0;
  summary.queue.producer_wait_ms = 0.0;
  summary.timing.processing_wall_ms = 10.0;
  summary.timing.includes = {"queue", "inference", "item JSON", "join"};
  summary.timing.excludes = {"discovery", "session construction"};
  summary.latency_ms.sample_count = 1;
  summary.latency_ms.mean_ms = 8.0;
  summary.latency_ms.p50_ms = 8.0;
  summary.latency_ms.p95_ms = 8.0;
  summary.throughput_images_per_second = 100.0;
  summary.memory.supported = true;
  summary.memory.status = "supported";
  summary.memory.metric = "peak_rss";
  summary.memory.bytes = 1024;
  summary.memory.mebibytes = 1.0 / 1024.0;
  summary.memory.scope = "process lifetime";

  BatchItemResult item;
  item.sequence_index = 0;
  item.status = BatchItemStatus::kSucceeded;
  item.source_path = root / "input" / "image.jpg";
  item.json_output_path = root / "output" / "items" /
                          "000000.detections.json";
  item.detection_count = 3;
  item.latency_ms = 8.0;
  summary.items = {item};
  summary.limitations = {"test limitation"};
  return summary;
}

RuntimeContract make_fake_runtime_contract(
    const std::filesystem::path& root) {
  RuntimeContract contract;
  contract.runtime.schema_version = 1;
  contract.runtime.declaration_path = root / "config.txt";
  contract.runtime.artifact_spec_path = root / "artifact.txt";
  contract.runtime.score_threshold = 0.25;
  contract.runtime.nms_threshold = 0.45;
  contract.runtime.provider = ExecutionProvider::kCpu;

  contract.artifact.schema_version = 1;
  contract.artifact.declaration_path = root / "artifact.txt";
  contract.artifact.model_id = "fake-yolov8";
  contract.artifact.model_family = ModelFamily::kYoloV8;
  contract.artifact.model_path = root / "model.onnx";
  contract.artifact.model_sha256 = std::string(64, 'A');
  contract.artifact.opset = 17;
  contract.artifact.input.name = "images";
  contract.artifact.input.shape = {1, 3, 800, 800};
  contract.artifact.input.dtype = TensorDataType::kFloat32;
  contract.artifact.input.layout = TensorLayout::kNchw;
  contract.artifact.output.name = "output0";
  contract.artifact.output.shape = {1, 10, 8400};
  contract.artifact.output.dtype = TensorDataType::kFloat32;
  contract.artifact.output.layout = TensorLayout::kBcn;
  contract.artifact.class_names = {"defect"};
  contract.artifact.nms_mode = NmsMode::kClassAgnostic;

  write_file(contract.runtime.declaration_path, "fake config");
  write_file(contract.artifact.declaration_path, "fake artifact");
  write_file(contract.artifact.model_path, "fake model");
  return contract;
}

std::filesystem::path make_fake_image_directory(
    const std::filesystem::path& root, std::size_t count) {
  const std::filesystem::path input = root / "input";
  for (std::size_t index = 0; index < count; ++index) {
    std::ostringstream filename;
    filename << std::setw(3) << std::setfill('0') << index << ".jpg";
    write_file(input / filename.str(), "fake image bytes");
  }
  return input;
}

BatchRequest make_fake_batch_request(
    const std::filesystem::path& root,
    const std::filesystem::path& input,
    std::size_t workers,
    std::size_t queue_capacity) {
  BatchRequest request;
  request.input_kind = BatchInputKind::kDirectory;
  request.input_path = input;
  request.output_directory = root / "output";
  request.summary_path = root / "summary.json";
  request.requested_workers = workers;
  request.queue_capacity = queue_capacity;
  request.command_arguments = {"fake_batch_test", "--batch"};
  return request;
}

ModelMetadata make_fake_model_metadata() {
  ModelMetadata metadata;
  metadata.ort_version = "1.19.2";
  metadata.session_provider = "CPUExecutionProvider";
  metadata.provider_evidence = "fake executor CPU provider evidence";
  metadata.intra_op_num_threads = 1;
  metadata.inter_op_num_threads = 1;
  metadata.execution_mode = "sequential";
  metadata.graph_optimization_level = "all";
  return metadata;
}

struct FakeExecutorState {
  explicit FakeExecutorState(std::size_t task_count)
      : calls(task_count, 0),
        started(task_count, false),
        finished(task_count, false),
        completion_allowed(task_count, false) {}

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<std::size_t> calls;
  std::vector<bool> started;
  std::vector<bool> finished;
  std::vector<bool> completion_allowed;
  std::vector<std::size_t> completion_order;
  std::unordered_set<std::size_t> failing_tasks;
  std::optional<std::size_t> failing_initialization_worker;
  bool gate_completion = false;
  std::size_t created_executors = 0;
  std::size_t destroyed_executors = 0;
  std::size_t active_calls = 0;
};

class FakeBatchTaskExecutor final : public internal::BatchTaskExecutor {
 public:
  FakeBatchTaskExecutor(std::shared_ptr<FakeExecutorState> state,
                        ModelMetadata metadata,
                        std::size_t worker_index)
      : state_(std::move(state)),
        metadata_(std::move(metadata)),
        worker_index_(worker_index) {}

  ~FakeBatchTaskExecutor() override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ++state_->destroyed_executors;
    state_->condition.notify_all();
  }

  const ModelMetadata& metadata() const override { return metadata_; }

  double session_initialization_ms() const override {
    return 1.0 + static_cast<double>(worker_index_);
  }

  internal::BatchExecutionResult run(
      const std::filesystem::path& source_path,
      const DetectionOutputRequest& output_request) override {
    const std::size_t task_index = static_cast<std::size_t>(
        std::stoull(source_path.stem().string()));
    bool should_fail = false;
    {
      std::unique_lock<std::mutex> lock(state_->mutex);
      if (task_index >= state_->calls.size()) {
        throw std::runtime_error("fake executor received unknown task index");
      }
      ++state_->calls[task_index];
      state_->started[task_index] = true;
      ++state_->active_calls;
      state_->condition.notify_all();
      if (state_->gate_completion) {
        state_->condition.wait(lock, [&] {
          return state_->completion_allowed[task_index];
        });
      }
      should_fail =
          state_->failing_tasks.find(task_index) !=
          state_->failing_tasks.end();
    }

    // Guarantees a positive item latency even on fast test machines.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->finished[task_index] = true;
      state_->completion_order.push_back(task_index);
      --state_->active_calls;
      state_->condition.notify_all();
    }
    if (should_fail) {
      throw std::runtime_error(
          "fake per-image failure for task " +
          std::to_string(task_index));
    }

    internal::BatchExecutionResult result;
    result.detection_count = task_index + 1;
    result.outputs.json_path = output_request.json_path;
    result.outputs.image_path = output_request.image_path;
    return result;
  }

 private:
  std::shared_ptr<FakeExecutorState> state_;
  ModelMetadata metadata_;
  std::size_t worker_index_ = 0;
};

class FakeBatchExecutorFactory final
    : public internal::BatchExecutorFactory {
 public:
  explicit FakeBatchExecutorFactory(
      std::shared_ptr<FakeExecutorState> state)
      : state_(std::move(state)), metadata_(make_fake_model_metadata()) {}

  std::unique_ptr<internal::BatchTaskExecutor> create(
      const RuntimeContract& /*contract*/,
      std::size_t worker_index) override {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->failing_initialization_worker.has_value() &&
        *state_->failing_initialization_worker == worker_index) {
      throw std::runtime_error(
          "fake executor initialization failure for worker " +
          std::to_string(worker_index));
    }
    ++state_->created_executors;
    return std::make_unique<FakeBatchTaskExecutor>(
        state_, metadata_, worker_index);
  }

 private:
  std::shared_ptr<FakeExecutorState> state_;
  ModelMetadata metadata_;
};

template <typename Predicate>
bool wait_for_fake_state(const std::shared_ptr<FakeExecutorState>& state,
                         Predicate&& predicate) {
  std::unique_lock<std::mutex> lock(state->mutex);
  return state->condition.wait_for(
      lock, std::chrono::seconds(2),
      [&] { return predicate(*state); });
}

void allow_fake_task(const std::shared_ptr<FakeExecutorState>& state,
                     std::size_t task_index) {
  std::lock_guard<std::mutex> lock(state->mutex);
  state->completion_allowed[task_index] = true;
  state->condition.notify_all();
}

void allow_all_fake_tasks(
    const std::shared_ptr<FakeExecutorState>& state) {
  std::lock_guard<std::mutex> lock(state->mutex);
  std::fill(state->completion_allowed.begin(),
            state->completion_allowed.end(), true);
  state->condition.notify_all();
}

TEST(BatchDiscoveryDirectoryTest, RecursesFiltersAndSortsGenericPaths) {
  TemporaryDirectory temporary("directory_discovery");
  const std::filesystem::path input = temporary.path() / "input";
  write_file(input / "02.jpg");
  write_file(input / "nested" / "01.PNG");
  write_file(input / "nested" / "03.tIfF");
  write_file(input / "ignored.txt");
  write_file(input / "looks-like-image.jpg.tmp");

  const std::vector<BatchTask> tasks =
      discover_batch_tasks(BatchInputKind::kDirectory, input);

  ASSERT_EQ(tasks.size(), 3U);
  EXPECT_EQ(task_logical_paths(tasks),
            (std::vector<std::filesystem::path>{
                "02.jpg", "nested/01.PNG", "nested/03.tIfF"}));
  for (std::size_t index = 0; index < tasks.size(); ++index) {
    EXPECT_EQ(tasks[index].sequence_index, index);
    EXPECT_TRUE(tasks[index].source_path.is_absolute());
  }
}

TEST(BatchDiscoveryDirectoryTest, DoesNotFollowDirectorySymlinksWhenAvailable) {
  TemporaryDirectory temporary("directory_symlink");
  const std::filesystem::path input = temporary.path() / "input";
  const std::filesystem::path outside = temporary.path() / "outside";
  write_file(input / "kept.jpg");
  write_file(outside / "must-not-be-followed.jpg");

  std::error_code symlink_error;
  std::filesystem::create_directory_symlink(
      outside, input / "linked-directory", symlink_error);

  const std::vector<BatchTask> tasks =
      discover_batch_tasks(BatchInputKind::kDirectory, input);
  ASSERT_EQ(tasks.size(), 1U);
  EXPECT_EQ(tasks.front().logical_path,
            std::filesystem::path("kept.jpg"));
  if (symlink_error) {
    GTEST_LOG_(INFO) << "Directory symlink unavailable on this platform: "
                     << symlink_error.message();
  }
}

TEST(BatchDiscoveryDirectoryTest,
     RejectsSymlinkOrReparsePointAsInputRoot) {
  TemporaryDirectory temporary("directory_root_symlink");
  const std::filesystem::path real_input = temporary.path() / "real-input";
  write_file(real_input / "image.jpg");
  const std::filesystem::path linked_input =
      temporary.path() / "linked-input";
  std::error_code symlink_error;
  std::filesystem::create_directory_symlink(
      real_input, linked_input, symlink_error);
  if (symlink_error) {
    GTEST_SKIP() << "Directory symlink unavailable on this platform: "
                 << symlink_error.message();
  }

  const std::string message = capture_runtime_error([&] {
    (void)discover_batch_tasks(
        BatchInputKind::kDirectory, linked_input);
  });

  EXPECT_NE(message.find("without symlink/reparse indirection"),
            std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(BatchDiscoveryDirectoryTest, RejectsAnEmptyImageSet) {
  TemporaryDirectory temporary("empty_directory");
  const std::filesystem::path input = temporary.path() / "input";
  write_file(input / "ignored.txt");

  const std::string message = capture_runtime_error([&] {
    (void)discover_batch_tasks(BatchInputKind::kDirectory, input);
  });
  EXPECT_NE(message.find("at least one supported regular image file"),
            std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(BatchDiscoveryManifestTest, AcceptsBomCrLfCommentsAndPreservesOrder) {
  TemporaryDirectory temporary("manifest_order");
  const std::filesystem::path image_a =
      temporary.path() / "images" / "a.jpg";
  const std::filesystem::path image_b =
      temporary.path() / "images" / "b.png";
  write_file(image_a);
  write_file(image_b);
  const std::filesystem::path manifest = temporary.path() / "inputs.txt";
  write_file(
      manifest,
      std::string("\xEF\xBB\xBF") +
          "   # UTF-8 BOM and leading-space comment\r\n"
          "images/b.png\r\n"
          "\r\n"
          "# another comment\r\n"
          "images/a.jpg\r\n");

  const std::vector<BatchTask> tasks =
      discover_batch_tasks(BatchInputKind::kManifest, manifest);

  ASSERT_EQ(tasks.size(), 2U);
  EXPECT_EQ(task_logical_paths(tasks),
            (std::vector<std::filesystem::path>{
                "images/b.png", "images/a.jpg"}));
  EXPECT_EQ(tasks[0].source_path, std::filesystem::canonical(image_b));
  EXPECT_EQ(tasks[1].source_path, std::filesystem::canonical(image_a));
}

TEST(BatchDiscoveryManifestTest, RejectsAbsoluteMissingAndUnsupportedPaths) {
  TemporaryDirectory temporary("manifest_errors");
  const std::filesystem::path image = temporary.path() / "image.jpg";
  const std::filesystem::path text = temporary.path() / "not-image.txt";
  const std::filesystem::path manifest = temporary.path() / "inputs.txt";
  write_file(image);
  write_file(text);

  write_file(manifest, image.generic_string() + "\n");
  EXPECT_NE(capture_runtime_error([&] {
              (void)discover_batch_tasks(
                  BatchInputKind::kManifest, manifest);
            }).find("relative image path"),
            std::string::npos);

  write_file(manifest, "missing.jpg\n");
  EXPECT_NE(capture_runtime_error([&] {
              (void)discover_batch_tasks(
                  BatchInputKind::kManifest, manifest);
            }).find("existing accessible image file"),
            std::string::npos);

  write_file(manifest, "not-image.txt\n");
  EXPECT_NE(capture_runtime_error([&] {
              (void)discover_batch_tasks(
                  BatchInputKind::kManifest, manifest);
            }).find("ending in"),
            std::string::npos);
}

TEST(BatchDiscoveryManifestTest, RejectsDuplicateCanonicalInputs) {
  TemporaryDirectory temporary("manifest_duplicate");
  write_file(temporary.path() / "images" / "same.jpg");
  const std::filesystem::path manifest = temporary.path() / "inputs.txt";
  write_file(manifest,
             "images/same.jpg\nimages/../images/same.jpg\n");

  const std::string message = capture_runtime_error([&] {
    (void)discover_batch_tasks(BatchInputKind::kManifest, manifest);
  });
  EXPECT_NE(message.find("duplicate"), std::string::npos) << message;
  EXPECT_NE(message.find("action:"), std::string::npos) << message;
}

TEST(BatchDiscoveryManifestTest,
     CanonicalDuplicateKeyUsesPlatformUnicodeCaseSemantics) {
  TemporaryDirectory temporary("manifest_unicode_case");
  const std::filesystem::path uppercase_name =
      std::filesystem::u8path(u8"\u00C4Image.jpg");
  const std::filesystem::path lowercase_name =
      std::filesystem::u8path(u8"\u00E4image.jpg");
  write_file(temporary.path() / uppercase_name);
  const std::filesystem::path manifest = temporary.path() / "inputs.txt";
#ifdef _WIN32
  if (!std::filesystem::exists(temporary.path() / lowercase_name)) {
    GTEST_SKIP() << "The temporary Windows directory is explicitly "
                    "case-sensitive";
  }
  write_file(manifest, uppercase_name.generic_u8string() + "\n" +
                           lowercase_name.generic_u8string() + "\n");

  const std::string message = capture_runtime_error([&] {
    (void)discover_batch_tasks(BatchInputKind::kManifest, manifest);
  });
  EXPECT_NE(message.find("duplicate"), std::string::npos) << message;
  const internal::BatchPathLocationLess less;
  EXPECT_FALSE(less(uppercase_name, lowercase_name));
  EXPECT_FALSE(less(lowercase_name, uppercase_name));
#else
  write_file(temporary.path() / lowercase_name);
  write_file(manifest, uppercase_name.generic_u8string() + "\n" +
                           lowercase_name.generic_u8string() + "\n");

  const std::vector<BatchTask> tasks =
      discover_batch_tasks(BatchInputKind::kManifest, manifest);
  EXPECT_EQ(tasks.size(), 2U);
  const internal::BatchPathLocationLess less;
  EXPECT_TRUE(less(uppercase_name, lowercase_name) ||
              less(lowercase_name, uppercase_name));
#endif
}

TEST(BoundedQueueTest, RejectsZeroCapacity) {
  EXPECT_THROW((internal::BoundedQueue<int>(0)), std::invalid_argument);
}

TEST(BoundedQueueTest, PreservesFifoAndNeverExceedsCapacity) {
  internal::BoundedQueue<int> queue(2);
  EXPECT_TRUE(queue.push(10));
  EXPECT_TRUE(queue.push(20));
  EXPECT_EQ(queue.size(), 2U);
  EXPECT_EQ(queue.statistics().capacity, 2U);
  EXPECT_EQ(queue.statistics().peak_depth, 2U);
  const std::optional<int> first = queue.pop();
  const std::optional<int> second = queue.pop();
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*first, 10);
  EXPECT_EQ(*second, 20);
  queue.close();
  EXPECT_FALSE(queue.pop().has_value());
}

TEST(BoundedQueueTest, CloseRefusesProducersButDrainsExistingItems) {
  internal::BoundedQueue<int> queue(1);
  ASSERT_TRUE(queue.push(7));
  queue.close();

  EXPECT_TRUE(queue.closed());
  EXPECT_FALSE(queue.stopped());
  EXPECT_FALSE(queue.push(8));
  ASSERT_TRUE(queue.pop().has_value());
  EXPECT_FALSE(queue.pop().has_value());
}

TEST(BoundedQueueTest, StopAbandonsPendingItemsAndRefusesAllOperations) {
  internal::BoundedQueue<int> queue(2);
  ASSERT_TRUE(queue.push(1));
  ASSERT_TRUE(queue.push(2));
  queue.request_stop();

  EXPECT_TRUE(queue.closed());
  EXPECT_TRUE(queue.stopped());
  EXPECT_EQ(queue.size(), 0U);
  EXPECT_FALSE(queue.push(3));
  EXPECT_FALSE(queue.pop().has_value());
}

TEST(BoundedQueueTest, AppliesBackpressureAndRecordsProducerWait) {
  internal::BoundedQueue<int> queue(1);
  ASSERT_TRUE(queue.push(1));
  std::future<bool> producer = std::async(
      std::launch::async, [&queue] { return queue.push(2); });
  if (!wait_until([&queue] {
        return queue.statistics().producer_wait_count == 1U;
      })) {
    queue.request_stop();
    FAIL() << "Producer did not enter bounded-queue backpressure";
  }
  EXPECT_EQ(producer.wait_for(std::chrono::milliseconds(0)),
            std::future_status::timeout);

  const std::optional<int> first = queue.pop();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(*first, 1);
  ASSERT_TRUE(producer.get());
  const std::optional<int> second = queue.pop();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(*second, 2);
  const internal::BoundedQueueStatistics statistics = queue.statistics();
  EXPECT_EQ(statistics.capacity, 1U);
  EXPECT_EQ(statistics.peak_depth, 1U);
  EXPECT_EQ(statistics.producer_wait_count, 1U);
  EXPECT_GT(statistics.producer_wait_duration.count(), 0);
}

TEST(BoundedQueueTest, StopWakesABlockedProducer) {
  internal::BoundedQueue<int> queue(1);
  ASSERT_TRUE(queue.push(1));
  std::future<bool> producer = std::async(
      std::launch::async, [&queue] { return queue.push(2); });
  if (!wait_until([&queue] {
        return queue.statistics().producer_wait_count == 1U;
      })) {
    queue.request_stop();
    FAIL() << "Producer did not block before cooperative stop";
  }

  queue.request_stop();

  EXPECT_FALSE(producer.get());
  EXPECT_EQ(queue.size(), 0U);
}

TEST(BoundedQueueTest, StopWakesABlockedConsumer) {
  internal::BoundedQueue<int> queue(1);
  std::promise<void> consumer_started;
  std::future<void> started = consumer_started.get_future();
  std::future<std::optional<int>> consumer = std::async(
      std::launch::async, [&queue, &consumer_started] {
        consumer_started.set_value();
        return queue.pop();
      });
  started.wait();

  queue.request_stop();

  EXPECT_FALSE(consumer.get().has_value());
}

TEST(BatchRunnerPathSafetyTest,
     RejectsItemsDirectorySymlinkThatEscapesOutputRoot) {
  TemporaryDirectory temporary("items_symlink_escape");
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), 1);
  const BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 1, 1);
  std::filesystem::create_directories(request.output_directory);
  const std::filesystem::path outside = temporary.path() / "outside";
  std::filesystem::create_directories(outside);
  std::error_code symlink_error;
  std::filesystem::create_directory_symlink(
      outside, request.output_directory / "items", symlink_error);
  if (symlink_error) {
    GTEST_SKIP() << "Directory symlink unavailable on this platform: "
                 << symlink_error.message();
  }
  auto state = std::make_shared<FakeExecutorState>(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const std::string message = capture_runtime_error(
      [&] { (void)runner.run(request); });

  EXPECT_NE(message.find("output.item_directory"), std::string::npos)
      << message;
  EXPECT_NE(message.find("symlink"), std::string::npos) << message;
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->created_executors, 0U);
}

TEST(BatchRunnerProviderContractTest,
     RejectsNativeTensorRtBeforeConstructingAnyExecutor) {
  TemporaryDirectory temporary("native_provider_rejected");
  RuntimeContract contract = make_fake_runtime_contract(temporary.path());
  contract.runtime.schema_version = 2;
  contract.runtime.provider = ExecutionProvider::kTensorRtNative;
  auto state = std::make_shared<FakeExecutorState>(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);

  const std::string message = capture_runtime_error(
      [&] { BatchRunner runner(contract, factory); });

  EXPECT_NE(message.find("runtime.provider"), std::string::npos) << message;
  EXPECT_NE(message.find("expected cpu"), std::string::npos) << message;
  EXPECT_NE(message.find("actual tensorrt_native"), std::string::npos)
      << message;
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->created_executors, 0U);
}

TEST(BatchRunnerPathSafetyTest,
     ContainmentUsesPathComponentsRatherThanStringPrefixes) {
  const std::filesystem::path root =
      std::filesystem::path("root") / "output";
  EXPECT_TRUE(internal::batch_path_is_same_or_descendant(
      root / "items", root));
  EXPECT_TRUE(internal::batch_path_is_same_or_descendant(root, root));
  EXPECT_FALSE(internal::batch_path_is_same_or_descendant(
      std::filesystem::path("root") / "output-sibling" / "items", root));
}

TEST(BatchRunnerPathSafetyTest,
     RejectsItemsPathWhenItIsARegularOrSpecialObject) {
  TemporaryDirectory temporary("items_regular_file");
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), 1);
  const BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 1, 1);
  write_file(request.output_directory / "items", "not a directory");
  auto state = std::make_shared<FakeExecutorState>(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const std::string message = capture_runtime_error(
      [&] { (void)runner.run(request); });

  EXPECT_NE(message.find("output.item_directory"), std::string::npos)
      << message;
  EXPECT_NE(message.find("special object"), std::string::npos) << message;
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->created_executors, 0U);
}

#ifdef _WIN32
TEST(BatchRunnerPathSafetyTest,
     UnicodeOrdinalIgnoreCaseProtectsEquivalentWindowsInputPath) {
  TemporaryDirectory temporary("unicode_windows_protected_path");
  RuntimeContract contract = make_fake_runtime_contract(temporary.path());
  const std::filesystem::path uppercase_config =
      temporary.path() / std::filesystem::u8path(u8"\u00C4Config.txt");
  const std::filesystem::path lowercase_alias =
      temporary.path() / std::filesystem::u8path(u8"\u00E4config.txt");
  write_file(uppercase_config, "unicode config");
  contract.runtime.declaration_path = uppercase_config;
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), 1);
  BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 1, 1);
  request.summary_path = lowercase_alias;
  request.overwrite_existing = true;
  EXPECT_TRUE(internal::batch_path_text_equal(
      uppercase_config, lowercase_alias));
  auto state = std::make_shared<FakeExecutorState>(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const std::string message = capture_runtime_error(
      [&] { (void)runner.run(request); });

  EXPECT_NE(message.find("protected input"), std::string::npos) << message;
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->created_executors, 0U);
}
#else
TEST(BatchRunnerPathSafetyTest,
     PosixProtectedPathsRemainUnicodeCaseSensitive) {
  TemporaryDirectory temporary("unicode_posix_protected_path");
  RuntimeContract contract = make_fake_runtime_contract(temporary.path());
  const std::filesystem::path uppercase_config =
      temporary.path() / std::filesystem::u8path(u8"\u00C4Config.txt");
  const std::filesystem::path lowercase_distinct =
      temporary.path() / std::filesystem::u8path(u8"\u00E4config.txt");
  write_file(uppercase_config, "unicode config");
  contract.runtime.declaration_path = uppercase_config;
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), 1);
  BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 1, 1);
  request.summary_path = lowercase_distinct;
  EXPECT_FALSE(internal::batch_path_text_equal(
      uppercase_config, lowercase_distinct));
  auto state = std::make_shared<FakeExecutorState>(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const BatchSummary summary = runner.run(request);

  EXPECT_EQ(summary.status, BatchStatus::kSucceeded);
  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->created_executors, 1U);
  EXPECT_EQ(state->destroyed_executors, 1U);
}
#endif

TEST(BatchRunnerInjectedExecutorTest,
     ExecutesEachTaskOnceAndKeepsSummaryInDiscoveryOrder) {
  TemporaryDirectory temporary("injected_order");
  constexpr std::size_t kTaskCount = 3;
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), kTaskCount);
  const BatchRequest request = make_fake_batch_request(
      temporary.path(), input, kTaskCount, kTaskCount);
  auto state = std::make_shared<FakeExecutorState>(kTaskCount);
  state->gate_completion = true;
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  std::future<BatchSummary> future = std::async(
      std::launch::async, [&] { return runner.run(request); });
  if (!wait_for_fake_state(state, [](const FakeExecutorState& value) {
        return std::all_of(value.started.begin(), value.started.end(),
                           [](bool started) { return started; });
      })) {
    allow_all_fake_tasks(state);
    FAIL() << "All fake tasks did not start";
  }

  allow_fake_task(state, 2);
  if (!wait_for_fake_state(state, [](const FakeExecutorState& value) {
        return value.finished[2];
      })) {
    allow_all_fake_tasks(state);
    FAIL() << "Task 2 did not complete after release";
  }
  allow_fake_task(state, 0);
  if (!wait_for_fake_state(state, [](const FakeExecutorState& value) {
        return value.finished[0];
      })) {
    allow_all_fake_tasks(state);
    FAIL() << "Task 0 did not complete after release";
  }
  allow_fake_task(state, 1);

  const BatchSummary summary = future.get();
  ASSERT_EQ(summary.status, BatchStatus::kSucceeded);
  EXPECT_EQ(summary.counts.discovered, kTaskCount);
  EXPECT_EQ(summary.counts.started, kTaskCount);
  EXPECT_EQ(summary.counts.succeeded, kTaskCount);
  EXPECT_EQ(summary.counts.failed, 0U);
  EXPECT_EQ(summary.counts.cancelled, 0U);
  ASSERT_EQ(summary.items.size(), kTaskCount);
  for (std::size_t index = 0; index < kTaskCount; ++index) {
    EXPECT_EQ(summary.items[index].sequence_index, index);
    std::ostringstream expected_stem;
    expected_stem << std::setw(3) << std::setfill('0') << index;
    EXPECT_EQ(summary.items[index].source_path.stem().string(),
              expected_stem.str());
    EXPECT_EQ(summary.items[index].status, BatchItemStatus::kSucceeded);
    EXPECT_EQ(summary.items[index].detection_count, index + 1);
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->calls,
            (std::vector<std::size_t>{1, 1, 1}));
  EXPECT_EQ(state->completion_order,
            (std::vector<std::size_t>{2, 0, 1}));
  EXPECT_EQ(state->active_calls, 0U);
  EXPECT_EQ(state->created_executors, kTaskCount);
  EXPECT_EQ(state->destroyed_executors, kTaskCount);
}

TEST(BatchRunnerInjectedExecutorTest,
     PerImageFailureDoesNotStopRemainingTasks) {
  TemporaryDirectory temporary("injected_item_failure");
  constexpr std::size_t kTaskCount = 4;
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), kTaskCount);
  const BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 2, 2);
  auto state = std::make_shared<FakeExecutorState>(kTaskCount);
  state->failing_tasks.insert(1);
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const BatchSummary summary = runner.run(request);

  EXPECT_EQ(summary.status, BatchStatus::kPartialFailure);
  EXPECT_EQ(summary.counts.discovered, kTaskCount);
  EXPECT_EQ(summary.counts.started, kTaskCount);
  EXPECT_EQ(summary.counts.succeeded, kTaskCount - 1);
  EXPECT_EQ(summary.counts.failed, 1U);
  EXPECT_EQ(summary.counts.cancelled, 0U);
  ASSERT_EQ(summary.items.size(), kTaskCount);
  EXPECT_EQ(summary.items[1].status, BatchItemStatus::kFailed);
  EXPECT_NE(summary.items[1].error.find("fake per-image failure"),
            std::string::npos);
  EXPECT_FALSE(summary.items[1].json_output_path.has_value());
  EXPECT_EQ(summary.items[0].status, BatchItemStatus::kSucceeded);
  EXPECT_EQ(summary.items[2].status, BatchItemStatus::kSucceeded);
  EXPECT_EQ(summary.items[3].status, BatchItemStatus::kSucceeded);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->calls,
            (std::vector<std::size_t>{1, 1, 1, 1}));
  EXPECT_EQ(state->active_calls, 0U);
  EXPECT_EQ(state->destroyed_executors, 2U);
}

TEST(BatchRunnerInjectedExecutorTest,
     InitializationFailureIsFatalAndRunsNoTask) {
  TemporaryDirectory temporary("injected_initialization_failure");
  constexpr std::size_t kTaskCount = 3;
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), kTaskCount);
  const BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 2, 2);
  auto state = std::make_shared<FakeExecutorState>(kTaskCount);
  state->failing_initialization_worker = 1;
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  const BatchSummary summary = runner.run(request);

  EXPECT_EQ(summary.status, BatchStatus::kFatal);
  EXPECT_NE(summary.fatal_error.find("initialization failed"),
            std::string::npos);
  EXPECT_EQ(summary.runtime.effective_workers, 2U);
  EXPECT_EQ(summary.runtime.session_count, 1U);
  EXPECT_EQ(summary.counts.enqueued, 0U);
  EXPECT_EQ(summary.counts.started, 0U);
  EXPECT_EQ(summary.counts.succeeded, 0U);
  EXPECT_EQ(summary.counts.failed, 0U);
  EXPECT_EQ(summary.counts.cancelled, kTaskCount);
  for (const BatchItemResult& item : summary.items) {
    EXPECT_EQ(item.status, BatchItemStatus::kCancelled);
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->calls,
            (std::vector<std::size_t>{0, 0, 0}));
  EXPECT_EQ(state->created_executors, 1U);
  EXPECT_EQ(state->destroyed_executors, 1U);
  EXPECT_EQ(state->active_calls, 0U);
}

TEST(BatchRunnerInjectedExecutorTest,
     RequestStopCancelsUnstartedTasksButCompletesAndJoinsRunningTask) {
  TemporaryDirectory temporary("injected_stop");
  constexpr std::size_t kTaskCount = 5;
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), kTaskCount);
  const BatchRequest request = make_fake_batch_request(
      temporary.path(), input, 1, kTaskCount - 1);
  auto state = std::make_shared<FakeExecutorState>(kTaskCount);
  state->gate_completion = true;
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  std::future<BatchSummary> future = std::async(
      std::launch::async, [&] { return runner.run(request); });
  if (!wait_for_fake_state(state, [](const FakeExecutorState& value) {
        return value.started[0];
      })) {
    allow_all_fake_tasks(state);
    FAIL() << "First fake task did not start";
  }

  runner.request_stop();
  EXPECT_EQ(future.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout)
      << "request_stop must not terminate a running executor call";
  allow_fake_task(state, 0);

  const BatchSummary summary = future.get();
  EXPECT_EQ(summary.status, BatchStatus::kCancelled);
  EXPECT_TRUE(summary.cooperative_stop_requested);
  EXPECT_EQ(summary.counts.discovered, kTaskCount);
  EXPECT_EQ(summary.counts.started, 1U);
  EXPECT_EQ(summary.counts.succeeded, 1U);
  EXPECT_EQ(summary.counts.failed, 0U);
  EXPECT_EQ(summary.counts.cancelled, kTaskCount - 1);
  EXPECT_EQ(summary.items[0].status, BatchItemStatus::kSucceeded);
  for (std::size_t index = 1; index < kTaskCount; ++index) {
    EXPECT_EQ(summary.items[index].status, BatchItemStatus::kCancelled);
    EXPECT_EQ(summary.items[index].latency_ms, 0.0);
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->calls,
            (std::vector<std::size_t>{1, 0, 0, 0, 0}));
  EXPECT_EQ(state->completion_order,
            (std::vector<std::size_t>{0}));
  EXPECT_EQ(state->active_calls, 0U);
  EXPECT_EQ(state->created_executors, 1U);
  EXPECT_EQ(state->destroyed_executors, 1U);
}

TEST(BatchRunnerInjectedExecutorTest,
     RequestStopRemainsCancelledWhenEveryTaskAlreadyStarted) {
  TemporaryDirectory temporary("injected_stop_all_started");
  constexpr std::size_t kTaskCount = 1;
  const RuntimeContract contract =
      make_fake_runtime_contract(temporary.path());
  const std::filesystem::path input =
      make_fake_image_directory(temporary.path(), kTaskCount);
  const BatchRequest request =
      make_fake_batch_request(temporary.path(), input, 1, 1);
  auto state = std::make_shared<FakeExecutorState>(kTaskCount);
  state->gate_completion = true;
  auto factory = std::make_shared<FakeBatchExecutorFactory>(state);
  BatchRunner runner(contract, factory);

  std::future<BatchSummary> future = std::async(
      std::launch::async, [&] { return runner.run(request); });
  if (!wait_for_fake_state(state, [](const FakeExecutorState& value) {
        return value.started[0];
      })) {
    allow_all_fake_tasks(state);
    FAIL() << "The only fake task did not start";
  }

  runner.request_stop();
  EXPECT_EQ(future.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout)
      << "request_stop must let the active executor call finish";
  allow_fake_task(state, 0);

  const BatchSummary summary = future.get();
  EXPECT_EQ(summary.status, BatchStatus::kCancelled);
  EXPECT_TRUE(summary.cooperative_stop_requested);
  EXPECT_EQ(summary.counts.discovered, 1U);
  EXPECT_EQ(summary.counts.enqueued, 1U);
  EXPECT_EQ(summary.counts.started, 1U);
  EXPECT_EQ(summary.counts.succeeded, 1U);
  EXPECT_EQ(summary.counts.failed, 0U);
  EXPECT_EQ(summary.counts.cancelled, 0U);
  ASSERT_EQ(summary.items.size(), 1U);
  EXPECT_EQ(summary.items[0].status, BatchItemStatus::kSucceeded);

  std::lock_guard<std::mutex> lock(state->mutex);
  EXPECT_EQ(state->calls, (std::vector<std::size_t>{1}));
  EXPECT_EQ(state->completion_order, (std::vector<std::size_t>{0}));
  EXPECT_EQ(state->active_calls, 0U);
  EXPECT_EQ(state->created_executors, 1U);
  EXPECT_EQ(state->destroyed_executors, 1U);
}

TEST(BatchSummaryTest, SerializesStableMachineReadableFieldsAndEscapesText) {
  TemporaryDirectory temporary("summary_json");
  BatchSummary summary = make_valid_summary(temporary.path());
  summary.limitations = {"quote=\" slash=\\ newline=\n"};

  const std::string json = serialize_batch_summary_json(summary);

  EXPECT_NE(json.find("\"schema_version\": 1"), std::string::npos) << json;
  EXPECT_NE(json.find("\"status\": \"succeeded\""),
            std::string::npos) << json;
  EXPECT_NE(json.find("\"cooperative_stop_requested\": false"),
            std::string::npos) << json;
  EXPECT_NE(json.find("\"sequence_index\": 0"),
            std::string::npos) << json;
  EXPECT_NE(json.find("quote=\\\" slash=\\\\ newline=\\n"),
            std::string::npos) << json;
}

TEST(BatchSummaryTest, RejectsBrokenCountInvariantAndNonFiniteNumbers) {
  TemporaryDirectory temporary("summary_invalid");
  BatchSummary broken_counts = make_valid_summary(temporary.path());
  broken_counts.counts.succeeded = 0;
  const std::string count_error = capture_runtime_error(
      [&broken_counts] { validate_batch_summary(broken_counts); });
  EXPECT_NE(count_error.find("counts"), std::string::npos) << count_error;
  EXPECT_NE(count_error.find("action:"), std::string::npos) << count_error;

  BatchSummary non_finite = make_valid_summary(temporary.path());
  non_finite.items[0].latency_ms =
      std::numeric_limits<double>::quiet_NaN();
  const std::string number_error = capture_runtime_error(
      [&non_finite] { validate_batch_summary(non_finite); });
  EXPECT_NE(number_error.find("latency"), std::string::npos) << number_error;
  EXPECT_NE(number_error.find("finite"), std::string::npos) << number_error;

  BatchSummary unsupported_memory = make_valid_summary(temporary.path());
  unsupported_memory.memory.supported = false;
  unsupported_memory.memory.status = "unavailable";
  unsupported_memory.memory.bytes = 0;
  unsupported_memory.memory.mebibytes = 0.0;
  unsupported_memory.memory.reason = "synthetic query failure";
  unsupported_memory.memory.publishable = true;
  const std::string publishability_error = capture_runtime_error(
      [&unsupported_memory] { validate_batch_summary(unsupported_memory); });
  EXPECT_NE(publishability_error.find("memory.publishable"),
            std::string::npos) << publishability_error;
  unsupported_memory.memory.publishable = false;
  EXPECT_NO_THROW(validate_batch_summary(unsupported_memory));
}

TEST(BatchSummaryTest, RefusesExistingFileUnlessOverwriteIsExplicit) {
  TemporaryDirectory temporary("summary_overwrite");
  BatchSummary summary = make_valid_summary(temporary.path());
  const std::filesystem::path output = temporary.path() / "summary.json";

  write_batch_summary_json(summary, output);
  const std::uintmax_t original_size = std::filesystem::file_size(output);
  const std::string refusal = capture_runtime_error(
      [&summary, &output] { write_batch_summary_json(summary, output); });
  EXPECT_NE(refusal.find("already exists"), std::string::npos) << refusal;
  EXPECT_EQ(std::filesystem::file_size(output), original_size);

  summary.timestamp_utc = "2026-08-30T00:00:01Z";
  EXPECT_NO_THROW(write_batch_summary_json(summary, output, true));
  std::ifstream input(output, std::ios::binary);
  const std::string rewritten((std::istreambuf_iterator<char>(input)),
                              std::istreambuf_iterator<char>());
  EXPECT_NE(rewritten.find("2026-08-30T00:00:01Z"), std::string::npos);
}

}  // namespace
}  // namespace yolo_defect_cpp
