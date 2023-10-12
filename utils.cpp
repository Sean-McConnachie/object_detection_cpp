#include "utils.h"

std::vector<std::string> list_dir(const std::string &path) {
    std::vector<std::string> paths;
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        paths.push_back(entry.path());
    }
    return paths;
}
