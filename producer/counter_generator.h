#pragma once

#include <atomic>
#include <cstdint>

#include "generator.h"

class CounterGenerator : public Generator<uint64_t> {
 public:
  explicit CounterGenerator(uint64_t start) : counter_(start) {}

  uint64_t Next() override { return counter_.fetch_add(1); }

  uint64_t Last() override { return counter_.load() - 1; }

  void Set(uint64_t start) { counter_.store(start); }

 private:
  std::atomic<uint64_t> counter_;
};
