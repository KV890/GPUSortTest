//
// Created by d306 on 9/6/23.
//
#pragma once

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#define CHECK(call)                                           \
  do {                                                        \
    cudaError_t error = call;                                 \
    if (error != cudaSuccess) {                               \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(error));                      \
      exit(1);                                                \
    }                                                         \
  } while (0)

#include <string>

constexpr int keySize = 16;
constexpr int valueSize = 32;

class KeyValue {
 public:
  __device__ __host__ KeyValue() : key{}, value{} {};

  char key[keySize + 1];  // +1防止越界
  char value[valueSize + 1];

  uint64_t sequence;

  __device__ __host__ bool operator<(const KeyValue& other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return static_cast<int>(p1[i]) < static_cast<int>(p2[i]);
      }
    }

    return sequence > other.sequence;
  }

  __device__ __host__ bool operator>(const KeyValue& other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return static_cast<int>(p1[i]) > static_cast<int>(p2[i]);
      }
    }

    return sequence < other.sequence;
  }

  __host__ __device__ bool operator==(const KeyValue &other) const {
    const auto *p1 = (const unsigned char *)key;
    const auto *p2 = (const unsigned char *)other.key;

    for (size_t i = 0; i < keySize; ++i) {
      if (static_cast<int>(p1[i]) != static_cast<int>(p2[i])) {
        return false;
      }
    }

    return true;
  }

};
