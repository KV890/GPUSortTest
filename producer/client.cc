//
// Created by d306 on 9/6/23.
//

#include "client.h"

int Client::DoInsert(int workload_type, size_t num) {
  std::string key;
  std::string val;
  if (workload_type == 0) {
    key = GenerateKeyFromInt(num);
    val = gen.Generate();
    sorting_factory.Insert(key, val);
  } else {
    key = workload.NextSequenceKey();
    workload.BuildValues(val);
    sorting_factory.Insert(key, val);
  }

  return 1;
}
