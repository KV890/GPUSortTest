//
// Created by d306 on 9/6/23.
//

#pragma once

#include <algorithm>
#include <vector>

#include "format.cuh"

// CPU 排序算法1，使用std库中的sort函数和unique函数完成排序和去重
size_t CPUSort1(std::vector<KeyValue>& kvs);

// 其他CPU排序算法，使用std::vector
size_t CPUSort2(std::vector<KeyValue>& kvs);

// 使用指针
size_t CPUSort3(KeyValue* kvs, size_t num_element);
