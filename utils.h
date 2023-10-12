#pragma once

#include <string>
#include <vector>
#include <random>
#include <filesystem>

#include "image.h"
#include "constants.h"

typedef std::vector<std::string> paths;

paths
list_dir(const std::string &path);

Img<float>
open_face(const std::string &path);

Img<float>
merge_images(const std::vector<Img<float>> &images);

std::vector<Img<float>>
sample_ims(const paths &ims, size_t n);