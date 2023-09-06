//
// Created by d306 on 9/6/23.
//

#include "Workload2.h"

#include <memory>

Random64 ran(1693992224830900 + 1);

std::string GenerateKeyFromInt(uint64_t num) {

  std::vector<std::unique_ptr<KeyGenerator>> key_gens(1);

  key_gens[0] = std::make_unique<KeyGenerator>(&ran, num, num);

  int64_t rand_num = key_gens[0]->Next();

  std::string key;
  key.resize(16);

  char* start = const_cast<char*>(key.data());
  char* pos = start;

  int bytes_to_fill = std::min(16 - static_cast<int>(pos - start), 8);
  for (int i = 0; i < bytes_to_fill; ++i) {
    pos[i] = (rand_num >> ((bytes_to_fill - i - 1) << 3)) & 0xFF;
  }
  pos += bytes_to_fill;
  if (16 > pos - start) {
    memset(pos, '0', 16 - (pos - start));
  }

  return key;
}
