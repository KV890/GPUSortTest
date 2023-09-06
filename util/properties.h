//
// Created by d306 on 9/6/23.
//
#pragma once

#include <map>
#include <string>

class Properties {
 public:
  std::string GetProperty(
      const std::string &key,
      const std::string &default_value = std::string()) const;

  const std::string &operator[](const std::string &key) const;

  const std::map<std::string, std::string> &properties() const;

  void SetProperty(const std::string &key, const std::string &value);

  bool Load(std::ifstream &input);

 private:
  std::map<std::string, std::string> properties_;
};
