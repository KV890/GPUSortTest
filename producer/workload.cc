//
// Created by d306 on 9/6/23.
//

#include "workload.h"

#include "../util/util.h"

std::string Workload::NextSequenceKey() const {
  uint64_t key_num = key_generator_->Next();
  return BuildKeyName(key_num);
}

std::string Workload::BuildKeyName(uint64_t key_num) {
  key_num = Hash(key_num);

  std::string key_str = "user";
  key_str += std::to_string(key_num);

  return key_str.substr(0, 16);
}

void Workload::BuildValues(std::string& value) const {
  value.append(field_len_generator_->Next(), RandomPrintChar());
}

void Workload::Init(const Properties& props) {
  num_ = std::stoull(props.GetProperty("num"));

  insert_key_sequence_.Set(num_);
  key_generator_ = new CounterGenerator(0);
  field_len_generator_ = new CounterGenerator(32);
}
