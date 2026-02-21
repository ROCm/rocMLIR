//===- ConcurrentQueue.h - Rate-adaptive MPMC queue -------------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H
#define ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H

#include "llvm/Support/Compiler.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace rocmlir::tuningdriver {

template <typename T>
class ConcurrentQueue {
public:
  // If maxCapacity is 0, the queue is unbounded
  explicit ConcurrentQueue(size_t maxCapacity = 0) : maxCapacity(maxCapacity) {}

  template <typename U>
  bool push(U &&item) {
    if (LLVM_UNLIKELY(done.load(std::memory_order_relaxed)))
      return false; // Early exit if terminated

    {
      std::unique_lock<std::mutex> lock(mtx);

      if (maxCapacity > 0) {
        cvNotFull.wait(lock, [this] {
          return queue.size() < currentCapacity ||
                 done.load(std::memory_order_relaxed);
        });
      }

      if (LLVM_UNLIKELY(done.load(std::memory_order_relaxed)))
        return false;

      queue.emplace(std::forward<U>(item));
    }

    cvNotEmpty.notify_one();
    return true;
  }

  bool pop(T &item) {
    std::unique_lock<std::mutex> lock(mtx);

    bool starved = queue.empty();
    cvNotEmpty.wait(lock, [this] {
      return !queue.empty() || done.load(std::memory_order_relaxed);
    });

    if (LLVM_UNLIKELY(queue.empty()))
      return false;

    item = std::move(queue.front());
    queue.pop();

    if (maxCapacity > 0) {
      if (starved) {
        // If the queue was empty, increase the capacity
        currentCapacity = std::min(currentCapacity + 1, maxCapacity);
        consecutiveFed = 0;
      } else {
        ++consecutiveFed;
        if (consecutiveFed >= fedShrinkThreshold) {
          // Decrease the capacity if the queue has been fed for a while
          currentCapacity = std::max(currentCapacity / 2, minCapacity);
          consecutiveFed = 0;
        }
      }
    }

    lock.unlock();

    cvNotFull.notify_one();
    return true;
  }

  void terminate() {
    done.store(true, std::memory_order_relaxed);
    cvNotEmpty.notify_all();
    cvNotFull.notify_all();
  }

  bool isTerminated() const { return done.load(std::memory_order_relaxed); }

private:
  static constexpr size_t minCapacity = 2;
  static constexpr size_t fedShrinkThreshold = 4;

  const size_t maxCapacity;
  size_t currentCapacity{maxCapacity};
  size_t consecutiveFed{0};

  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable cvNotEmpty;
  std::condition_variable cvNotFull;
  std::atomic<bool> done{false};
};

} // namespace rocmlir::tuningdriver

#endif // ROCMLIR_TUNING_DRIVER_CONCURRENT_QUEUE_H
