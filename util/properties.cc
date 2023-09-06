//
// Created by d306 on 9/6/23.
//
#include "properties.h"

#include <fstream>
#include <iostream>

#include "util.h"

std::string Properties::GetProperty(const std::string& key,
                                    const std::string& default_value) const {
  auto it = properties_.find(key);
  if (properties_.end() == it) {
    return default_value;
  } else
    return it->second;
}

const std::string& Properties::operator[](const std::string& key) const {
  return properties_.at(key);
}

const std::map<std::string, std::string>& Properties::properties() const {
  return properties_;
}

void Properties::SetProperty(const std::string& key, const std::string& value) {
  properties_[key] = value;
}

bool Properties::Load(std::ifstream& input) {
  if (!input.is_open()) {
    std::cerr << "File not open!" << std::endl;
    exit(1);
  }

  while (!input.eof() && !input.bad()) {
    std::string line;
    std::getline(input, line);
    if (line[0] == '#') continue;
    size_t pos = line.find_first_of('=');
    if (pos == std::string::npos) continue;
    SetProperty(Trim(line.substr(0, pos)), Trim(line.substr(pos + 1)));
  }
  return true;
}
