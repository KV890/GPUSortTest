//
// Created by d306 on 9/6/23.
//

#include "sorting_factory.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>

#include "cpu_sort.h"
#include "gpu_sort.cuh"

void SortingFactory::Insert(const std::string& key, const std::string& value) {
  KeyValue kv;
  memcpy(kv.key, key.c_str(), keySize);
  memcpy(kv.value, value.c_str(), valueSize);
  kv.sequence = kvs.size() + 1;

  kvs.emplace_back(kv);
}

void SortingFactory::Sort() const {
  std::vector<double> sorting_time;  // 记录排序时间

  std::vector<KeyValue> kvs_tmp1 = kvs;

  // GPU1 算法
  std::cout << "The size before sorting: " << kvs_tmp1.size() << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  KeyValue* sorted_kvs;

  size_t sorted_size = GPUSort3(kvs_tmp1.data(), kvs_tmp1.size(), &sorted_kvs);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);

  sorting_time.emplace_back(duration.count());

  std::cout << "The size after sorting: " << sorted_size
            << ", GPU1 sort time: " << duration.count() << " us"
            << ", " << static_cast<double>(duration.count()) / 1000000 << " sec"
            << std::endl;

  Assert2(sorted_kvs, sorted_size);

  // CPU1 算法
  std::vector<KeyValue> kvs_tmp2 = kvs;

  start_time = std::chrono::high_resolution_clock::now();

  CPUSort1(kvs_tmp2);

  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                   start_time);

  sorting_time.emplace_back(duration.count());

  std::cout << "The size after sorting: " << kvs_tmp2.size()
            << ", CPU1 sort time: " << duration.count() << " us"
            << ", " << static_cast<double>(duration.count()) / 1000000 << " sec"
            << std::endl;

  Assert(kvs_tmp2);

  std::cout << "Algorithm 1 is " << sorting_time[1] / sorting_time[0]
            << " times faster than algorithm 2" << std::endl;
}

void SortingFactory::Assert(std::vector<KeyValue> kvs_tmp) {
  for (size_t i = 1; i < kvs_tmp.size(); ++i) {
    if (kvs_tmp[i - 1] < kvs_tmp[i]) {
      // 正确
    } else {
      std::cerr << "Sort error!" << std::endl;

      std::cerr << "last string: " << kvs_tmp[i - 1].key
                << "sequence: " << kvs_tmp[i - 1].sequence << std::endl;
      std::cerr << "next string: " << kvs_tmp[i].key
                << "sequence: " << kvs_tmp[i].sequence << std::endl;

      exit(1);
    }
  }
}

void SortingFactory::Assert2(KeyValue* kvs_tmp, size_t sorted_size) {
  for (size_t i = 1; i < sorted_size; ++i) {
    if (kvs_tmp[i - 1] < kvs_tmp[i]) {
      // 正确
    } else {
      std::cerr << "Sort error!" << std::endl;

      std::cerr << "last string: " << kvs_tmp[i - 1].key
                << "sequence: " << kvs_tmp[i - 1].sequence << std::endl;
      std::cerr << "next string: " << kvs_tmp[i].key
                << "sequence: " << kvs_tmp[i].sequence << std::endl;

      exit(1);
    }
  }
}
