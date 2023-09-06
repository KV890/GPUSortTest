//
// Created by d306 on 9/6/23.
//

#include "cpu_sort.h"

size_t CPUSort1(std::vector<KeyValue>& kvs) {
  std::sort(kvs.begin(), kvs.end());
  kvs.erase(std::unique(kvs.begin(), kvs.end()), kvs.end());

  return kvs.size();
}

size_t CPUSort2(std::vector<KeyValue>& kvs) {

}

size_t CPUSort3(KeyValue* kvs, size_t num_element) {

}
