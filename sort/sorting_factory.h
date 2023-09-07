//
// Created by d306 on 9/6/23.
//
#pragma once

#include <string>
#include <vector>

#include "format.cuh"

class SortingFactory {
 public:
  void Insert(const std::string& key, const std::string& value);

  void Sort() const;

  // 判断排序结果是否正确
  // 排序结果仅当 字符串按照字典序排序，没有重复的元素时 正确
  static void Assert(std::vector<KeyValue> kvs_tmp);
  static void Assert2(KeyValue* kvs_tmp, size_t sorted_size);

  std::vector<KeyValue> kvs;
};
