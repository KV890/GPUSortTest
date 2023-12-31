#pragma once

#include <atomic>
#include <mutex>
#include <random>

#include "generator.h"

class UniformGenerator : public Generator<uint64_t> {
 public:
  // Both min and max are inclusive
  UniformGenerator(uint64_t min, uint64_t max) : dist_(min, max) { Next(); }

  uint64_t Next();
  uint64_t Last();

 private:
  std::mt19937_64 generator_;
  std::uniform_int_distribution<uint64_t> dist_;
  uint64_t last_int_;
  std::mutex mutex_;
};

inline uint64_t UniformGenerator::Next() {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_int_ = dist_(generator_);
}

inline uint64_t UniformGenerator::Last() {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_int_;
}
