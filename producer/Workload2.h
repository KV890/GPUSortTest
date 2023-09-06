//
// Created by d306 on 9/6/23.
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "../util/random.h"
#include "../util/util.h"

std::string GenerateKeyFromInt(uint64_t num);

inline std::string CompressibleString(Random* rnd, double compressed_fraction,
                                      int len, std::string* dst) {
  int raw = static_cast<int>(len * compressed_fraction);
  if (raw < 1) raw = 1;
  std::string raw_data = rnd->RandomString(raw);

  // Duplicate the random data until we have filled "len" bytes
  dst->clear();
  while (dst->size() < (unsigned int)len) {
    dst->append(raw_data);
  }
  dst->resize(len);
  return *dst;
}

class BaseDistribution {
 public:
  BaseDistribution(unsigned int _min, unsigned int _max)
      : min_value_size_(_min), max_value_size_(_max) {}
  virtual ~BaseDistribution() {}

  unsigned int Generate() {
    auto val = Get();
    if (NeedTruncate()) {
      val = std::max(min_value_size_, val);
      val = std::min(max_value_size_, val);
    }
    return val;
  }

 private:
  virtual unsigned int Get() = 0;
  virtual bool NeedTruncate() { return true; }
  unsigned int min_value_size_;
  unsigned int max_value_size_;
};

class FixedDistribution : public BaseDistribution {
 public:
  FixedDistribution(unsigned int size)
      : BaseDistribution(size, size), size_(size) {}

 private:
  virtual unsigned int Get() override { return size_; }
  virtual bool NeedTruncate() override { return false; }
  unsigned int size_;
};

class RandomGenerator {
 private:
  std::string data_;
  unsigned int pos_;
  std::unique_ptr<BaseDistribution> dist_;

 public:
  RandomGenerator() {
    auto max_value_size = 102400;
    dist_ = std::make_unique<FixedDistribution>(32);
    max_value_size = 32;
    // We use a limited amount of data over and over again and ensure
    // that it is larger than the compression window (32KB), and also
    // large enough to serve all typical value sizes we want to write.
    Random rnd(301);
    std::string piece;
    while (data_.size() < (unsigned)std::max(1048576, max_value_size)) {
      // Add a short fragment that is as compressible as specified
      // by FLAGS_compression_ratio.
      CompressibleString(&rnd, 0.5, 100, &piece);
      data_.append(piece);
    }
    pos_ = 0;
  }

  std::string Generate(unsigned int len) {
    assert(len <= data_.size());
    if (pos_ + len > data_.size()) {
      pos_ = 0;
    }

    std::string val(data_, pos_, len);
    pos_ += len;

    return val;
  }

  std::string Generate() {
    auto len = dist_->Generate();
    return Generate(len);
  }
};

class KeyGenerator {
 public:
  KeyGenerator(Random64* rand, uint64_t num,
               uint64_t /*num_per_set*/ = 64 * 1024)
      : rand_(rand), num_(num), next_(0) {}

  uint64_t Next() { return rand_->Next() % num_; }

  // Only available for UNIQUE_RANDOM mode.
  uint64_t Fetch(uint64_t index) {
    assert(index < values_.size());
    return values_[index];
  }

 private:
  Random64* rand_;
  const uint64_t num_;
  uint64_t next_;
  std::vector<uint64_t> values_;
};
