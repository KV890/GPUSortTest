//
// Created by d306 on 9/6/23.
//
#pragma once

#include <fstream>
#include <utility>

#include "../sort/sorting_factory.h"
#include "Workload2.h"
#include "workload.h"

class Client {
 public:
  Client(SortingFactory& _sorting_factory, RandomGenerator& _gen,
         Workload& _workload)
      : sorting_factory(_sorting_factory), gen(_gen), workload(_workload) {
  }

  int DoInsert(int workload_type, size_t num);

  ~Client() = default;

  SortingFactory& sorting_factory;
  Workload& workload;
  RandomGenerator& gen;
};

