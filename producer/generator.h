#pragma once

#include <cstdint>
#include <string>

template <typename Value>

class Generator {
 public:
  virtual Value Next() = 0;
  virtual Value Last() = 0;
  virtual ~Generator() {}
};
