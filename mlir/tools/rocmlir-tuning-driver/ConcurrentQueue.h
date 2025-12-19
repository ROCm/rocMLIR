//===- ConcurrentQueue.h - Simple MPMC queue --------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H
#define ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>

namespace rocmlir::tuningdriver {

template <typename T>
class ConcurrentQueue {
public:
  template <typename U>
  void push(U &&item) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      queue.push(std::forward<U>(item));
    }
    cv.notify_one();
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return !queue.empty() || done; });

    if (queue.empty())
      return false;

    item = std::move(queue.front());
    queue.pop();
    return true;
  }

  bool tryPop(T &item) {
    std::lock_guard<std::mutex> lock(mtx);
    if (queue.empty())
      return false;

    item = std::move(queue.front());
    queue.pop();
    return true;
  }

  void terminate() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      done = true;
    }
    cv.notify_all();
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mtx);
    done = false;
    std::queue<T> empty;
    queue.swap(empty);
  }

  bool isTerminated() const {
    std::lock_guard<std::mutex> lock(mtx);
    return done;
  }

  bool isDone() const {
    std::lock_guard<std::mutex> lock(mtx);
    return done && queue.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return queue.empty();
  }

private:
  std::queue<T> queue;
  mutable std::mutex mtx;
  std::condition_variable cv;
  bool done = false;
};

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H