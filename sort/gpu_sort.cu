//
// Created by d306 on 9/6/23.
//

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "gpu_sort.cuh"

size_t GPUSort1(std::vector<KeyValue>& kvs) {
  thrust::device_vector<KeyValue> kvs_d(kvs);

  thrust::sort(kvs_d.begin(), kvs_d.end());

  kvs_d.erase(thrust::unique(kvs_d.begin(), kvs_d.end()), kvs_d.end());

  thrust::copy(kvs_d.begin(), kvs_d.end(), kvs.begin());

  kvs.resize(kvs_d.size());

  return kvs.size();
}

size_t GPUSort2(std::vector<KeyValue>& kvs) {
  thrust::device_vector<KeyValue> kvs_d(kvs);

  // ...
  return 0;
}

size_t GPUSort3(KeyValue* kvs, size_t num_element) {
  // 将数据发送到设备内存
  KeyValue* kvs_d;
  cudaMalloc(&kvs_d, num_element * sizeof(KeyValue));
  cudaMemcpyAsync(kvs_d, kvs, num_element * sizeof(KeyValue),
                  cudaMemcpyHostToDevice);

  thrust::sort(thrust::device, kvs_d, kvs_d + num_element);

  size_t sorted_size =
      thrust::unique(thrust::device, kvs_d, kvs_d + num_element) - kvs_d;

  cudaFree(kvs_d);  // 释放资源

  // 将排序结果复制回主机内存
  cudaMemcpyAsync(kvs, kvs_d, sorted_size * sizeof(KeyValue),
                  cudaMemcpyDeviceToHost);

  return sorted_size;
}

size_t GPUSort4(KeyValue* kvs, size_t num_element) {
  // 将数据发送到设备内存
  KeyValue* kvs_d;
  cudaMalloc(&kvs_d, num_element * sizeof(KeyValue));
  cudaMemcpyAsync(kvs_d, kvs, num_element * sizeof(KeyValue),
                  cudaMemcpyHostToDevice);

  // ...

  cudaFree(kvs_d);  // 释放资源

  // 将排序结果复制回主机内存
  // ...

  return 0;
}
