#include <cstring>
#include <iostream>

#include "producer/client.h"
#include "producer/workload.h"
#include "sort/sorting_factory.h"
#include "util/properties.h"

bool StrStartWith(const char *str, const char *pre);
void UsageMessage(const char *command);
void ParseCommandLine(int argc, const char *argv[], Properties &props);

int main(const int argc, const char *argv[]) {
  Properties props;

  ParseCommandLine(argc, argv, props);

  // 0 : db_bench, 1 : ycsb-c
  int workload_type = stoi(props.GetProperty("workload_type", "0"));
  size_t num = std::stoull(props.GetProperty("num", "0"));

  SortingFactory sorting_factory;
  sorting_factory.kvs.reserve(num);

  Workload w;
  w.Init(props);

  RandomGenerator gen;

  Client client(sorting_factory, gen, w);

  int oks = 0;
  int ops_stage = 100;

  for (size_t i = 0; i < num; ++i) {
    oks += client.DoInsert(workload_type, num);

    if (oks >= ops_stage) {
      if (ops_stage < 1000)
        ops_stage += 100;
      else if (ops_stage < 5000)
        ops_stage += 500;
      else if (ops_stage < 10000)
        ops_stage += 1000;
      else if (ops_stage < 50000)
        ops_stage += 5000;
      else if (ops_stage < 100000)
        ops_stage += 10000;
      else if (ops_stage < 500000)
        ops_stage += 50000;
      else
        ops_stage += 100000;
      std::cerr << "... finished " << oks << std::endl;
    }

  }

  sorting_factory.Sort();

  return 0;
}

bool StrStartWith(const char *str, const char *pre) {
  return strncmp(str, pre, strlen(pre)) == 0;
}

void UsageMessage(const char *command) {
  std::cout << "Usage: " << command << " [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -workload n: Select a workload (default: 0)"
            << std::endl;
  std::cout
      << "  -num n: Specifies the amount of data to sort (default: 1000000)"
      << std::endl;
}

void ParseCommandLine(int argc, const char *argv[], Properties &props) {
  int argindex = 1;
  while (argindex < argc && StrStartWith(argv[argindex], "-")) {
    if (strcmp(argv[argindex], "-workload_type") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("workload_type", argv[argindex]);
      argindex++;
    } else if (strcmp(argv[argindex], "-num") == 0) {
      argindex++;
      if (argindex >= argc) {
        UsageMessage(argv[0]);
        exit(0);
      }
      props.SetProperty("num", argv[argindex]);
      argindex++;
    } else {
      std::cout << "Unknown option '" << argv[argindex] << "'" << std::endl;
      exit(0);
    }
  }

  if (argindex == 1 || argindex != argc) {
    UsageMessage(argv[0]);
    exit(0);
  }
}
