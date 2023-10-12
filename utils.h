#pragma once

#include <string>
#include <vector>
#include <filesystem>

typedef std::vector<std::string> paths;

paths
list_dir(const std::string &path);