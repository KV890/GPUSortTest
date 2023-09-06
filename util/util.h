//
// Created by d306 on 9/6/23.
//

#pragma once

#include <algorithm>
#include <climits>
#include <random>
#include <thread>

#define STORAGE_DECL static thread_local

const uint64_t kFNVOffsetBasis64 = 0xCBF29CE484222325;
const uint64_t kFNVPrime64 = 1099511628211;

inline std::string Trim(const std::string &str) {
  auto front = std::find_if_not(str.begin(), str.end(),
                                [](int c) { return std::isspace(c); });
  return std::string(
      front,
      std::find_if_not(str.rbegin(), std::string::const_reverse_iterator(front),
                       [](int c) { return std::isspace(c); })
          .base());
}

inline double RandomDouble(double min = 0.0, double max = 1.0) {
  static std::default_random_engine generator;
  static std::uniform_real_distribution<double> uniform(min, max);
  return uniform(generator);
}

inline uint64_t FNVHash64(uint64_t val) {
  uint64_t hash = kFNVOffsetBasis64;

  for (int i = 0; i < 8; i++) {
    uint64_t octet = val & 0x00ff;
    val = val >> 8;

    hash = hash ^ octet;
    hash = hash * kFNVPrime64;
  }
  return hash;
}

inline uint64_t Hash(uint64_t val) { return FNVHash64(val); }

inline char RandomPrintChar() { return rand() % 94 + 33; }