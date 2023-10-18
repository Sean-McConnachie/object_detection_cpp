#pragma once

#include <string>
#include <vector>
#include <random>
#include <filesystem>

#include "image.h"
#include "constants.h"

typedef std::vector<std::string> paths;

typedef std::vector<ImgType> images;

typedef struct {
    images ims;
    std::vector<int> labels;
} Samples;

typedef struct {
    double mean;
    double std;
} Stats;

paths
list_dir(const std::string &path);

ImgType
open_face(const std::string &path);

ImgType
merge_images(const images &images);

std::vector<std::unique_ptr<std::string>>
sample_paths(const paths &ims, size_t n);

images
sample_faces(const paths &ims, size_t n);

images
sample_backgrounds(const paths &ims, size_t n, bool resize = true);

ImgType
random_crop(const ImgType &img);

ImgType
open_background(const std::string &path, bool resize = true);

Samples
sample_data(int n_faces, int n_bgs, const paths &faces, const paths &bgs);

Samples
sample_data(int n_faces, int n_bgs, const paths &faces, const paths &bgs, Stats stats);

Stats
compute_stats(const images &ims);
