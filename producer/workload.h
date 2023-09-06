//
// Created by d306 on 9/6/23.
//

#pragma once

#include <string>

#include "../util/properties.h"
#include "counter_generator.h"
#include "generator.h"
#include "uniform_generator.h"
#include "zipfian_generator.h"

class Workload {
 public:
  Workload()
      : field_len_generator_(nullptr),
        key_generator_(nullptr),
        insert_key_sequence_(3),
        num_(0) {}

  void Init(const Properties &props);

  std::string NextSequenceKey() const;

  void BuildValues(std::string &values) const;

  static std::string BuildKeyName(uint64_t key_num);

  ~Workload() {
    delete field_len_generator_;
    delete key_generator_;
  }

  Generator<uint64_t> *field_len_generator_;
  Generator<uint64_t> *key_generator_;
  CounterGenerator insert_key_sequence_;
  size_t num_;
};