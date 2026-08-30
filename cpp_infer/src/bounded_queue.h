#ifndef YOLO_DEFECT_CPP_BOUNDED_QUEUE_H_
#define YOLO_DEFECT_CPP_BOUNDED_QUEUE_H_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace yolo_defect_cpp {
namespace internal {

struct BoundedQueueStatistics {
  std::size_t capacity = 0;
  std::size_t peak_depth = 0;
  std::size_t producer_wait_count = 0;
  std::chrono::nanoseconds producer_wait_duration{0};
};

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
      throw std::invalid_argument(
          "BoundedQueue capacity must be greater than zero");
    }
  }

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  bool push(T value) {
    using Clock = std::chrono::steady_clock;
    std::unique_lock<std::mutex> lock(mutex_);
    bool waited = false;
    Clock::time_point wait_started;
    if (!closed_ && !stopped_ && queue_.size() >= capacity_) {
      waited = true;
      wait_started = Clock::now();
      ++statistics_.producer_wait_count;
    }
    producer_condition_.wait(lock, [this] {
      return stopped_ || closed_ || queue_.size() < capacity_;
    });
    if (waited) {
      statistics_.producer_wait_duration +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              Clock::now() - wait_started);
    }
    if (stopped_ || closed_) {
      return false;
    }
    queue_.push_back(std::move(value));
    if (queue_.size() > statistics_.peak_depth) {
      statistics_.peak_depth = queue_.size();
    }
    lock.unlock();
    consumer_condition_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    consumer_condition_.wait(lock, [this] {
      return stopped_ || !queue_.empty() || closed_;
    });
    if (stopped_ || queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    producer_condition_.notify_one();
    return value;
  }

  // Normal completion: consumers drain existing entries and then stop.
  void close() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    producer_condition_.notify_all();
    consumer_condition_.notify_all();
  }

  // Cooperative cancellation: pending entries are deliberately abandoned.
  // Batch results are initialized as cancelled, so indices need not be
  // returned from this operation.
  void request_stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
      closed_ = true;
      queue_.clear();
    }
    producer_condition_.notify_all();
    consumer_condition_.notify_all();
  }

  bool stopped() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
  }

  bool closed() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  std::size_t size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  BoundedQueueStatistics statistics() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    BoundedQueueStatistics result = statistics_;
    result.capacity = capacity_;
    return result;
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable producer_condition_;
  std::condition_variable consumer_condition_;
  std::deque<T> queue_;
  bool closed_ = false;
  bool stopped_ = false;
  BoundedQueueStatistics statistics_;
};

}  // namespace internal
}  // namespace yolo_defect_cpp

#endif  // YOLO_DEFECT_CPP_BOUNDED_QUEUE_H_
