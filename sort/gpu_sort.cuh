//
// Created by d306 on 9/6/23.
//

#include <vector>

#include "format.cuh"

// 使用NVIDIA的thrust并行库进行排序、去重
size_t GPUSort1(std::vector<KeyValue>& kvs);

// 使用NVIDIA的thrust并行库进行排序、去重，与sort1不同的方法
// 或者不使用thrust库进行排序
size_t GPUSort2(std::vector<KeyValue>& kvs);

// 使用指针并使用thrust库进行排序
size_t GPUSort3(KeyValue* kvs, size_t num_element, KeyValue** sorted_kvs);

// 不使用thrust库进行排序，或者使用thrust库的其他方法
size_t GPUSort4(KeyValue* kvs, size_t num_element);
