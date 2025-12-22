//===- ConcurrentQueue.h - Simple MPMC queue --------------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H
#define ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H

#include "llvm/Support/Compiler.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace rocmlir::tuningdriver {

template <typename T>
class ConcurrentQueue {
public:
  template <typename U>
  bool push(U &&item) {
    if (LLVM_UNLIKELY(done.load(std::memory_order_relaxed)))
      return false; // Early exit if terminated

    {
      std::lock_guard<std::mutex> lock(mtx);
      if (LLVM_UNLIKELY(done.load(std::memory_order_relaxed)))
        return false; // Double-check after acquiring the lock

      queue.emplace(std::forward<U>(item));
    }

    cv.notify_one();
    return true;
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] {
      return !queue.empty() || done.load(std::memory_order_relaxed);
    });

    if (LLVM_UNLIKELY(queue.empty()))
      return false;

    item = std::move(queue.front());
    queue.pop();
    return true;
  }

  void terminate() {
    done.store(true, std::memory_order_relaxed);
    cv.notify_all();
  }

  bool isTerminated() const { return done.load(std::memory_order_relaxed); }

private:
  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable cv;
  std::atomic<bool> done{false};
};

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H
