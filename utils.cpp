#include "utils.h"

std::vector<std::string>
list_dir(const std::string &path) {
    std::vector<std::string> paths;
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        paths.push_back(entry.path());
    }
    return paths;
}

ImgType
open_face(const std::string &path) {
    ImgType im(FEATURE_SIZE, FEATURE_SIZE);

    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    cv::Rect roi(0, FACES_CROP_TOP, image.cols, image.rows - FACES_CROP_TOP);
    cv::Mat croppedImage = image(roi);
    int min = std::min(croppedImage.cols, croppedImage.rows);
    cv::Rect fitRect((croppedImage.cols - min) / 2, (croppedImage.rows - min) / 2, min, min);
    cv::Mat fitImage = croppedImage(fitRect);

    im.loadGrayScale(fitImage);

    return im;
}

ImgType
merge_images(const std::vector<ImgType> &images) {
    int max_height = 0;
    int total_width = 0;
    for (const auto &image: images) {
        max_height = std::max(max_height, static_cast<int>(image.height));
        total_width += (int) image.width;
    }

    ImgType merged(max_height, total_width);

    // set all pixels to 0
    for (int y = 0; y < max_height; ++y) {
        for (int x = 0; x < total_width; ++x) {
            merged.arr[y][x] = 0;
        }
    }

    int curr_width = 0;
    for (const auto &im: images) {
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                merged.arr[y][x + curr_width] = im.arr[y][x];
            }
        }
        curr_width += (int) im.width;
    }

    return merged;
}

std::vector<std::string *>
sample_paths(const paths &ims, size_t n) {
    std::vector<std::string *> result;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, (int) ims.size() - 1);

    for (size_t i = 0; i < n; ++i) {
        int index = distrib(gen);
        result.push_back(new std::string(ims[index]));
    }

    return result;
}

images
sample_faces(const paths &ims, size_t n) {
    std::vector<std::string *> paths = sample_paths(ims, n);
    std::vector<ImgType> result;
    for (const auto &path: paths) {
        result.push_back(open_face(*path));
    }
    return result;
}

images
sample_backgrounds(const paths &ims, size_t n, bool resize) {
    std::vector<std::string *> paths = sample_paths(ims, n);
    images result;
    for (const auto &path: paths) {
        result.push_back(open_background(*path, resize));
    }
    return result;
}

ImgType
random_crop(const ImgType &img) {
    std::random_device rd;
    std::mt19937 gen(rd());

    int max_size = std::min(img.height, img.width);
    int size = std::uniform_int_distribution<>(FEATURE_SIZE, max_size)(gen);
    int max_width = img.width - size - 1;
    int max_height = img.height - size - 1;

    int left = max_width <= 1 ? 0 : std::uniform_int_distribution<>(0, max_width)(gen);
    int top = max_height <= 1 ? 0 : std::uniform_int_distribution<>(0, max_height)(gen);

    ImgType cropped(size, size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; x++) {
            cropped.arr[y][x] = img.arr[y + top][x + left];
        }
    }
    return cropped;
}

ImgType
open_background(const std::string &path, bool resize) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    ImgType im(image.rows, image.cols);
    im.loadGrayScale(image);
    ImgType cropped = random_crop(im);

    if (resize) {
        return cropped.resize(FEATURE_SIZE, FEATURE_SIZE);
    } else {
        return cropped;
    }

}

Samples
sample_data(int n_faces, int n_bgs, const paths &faces, const paths &bgs) {
    Samples samples;

    for (auto &face: sample_faces(faces, n_faces)) {
        face.normalize(1.0);
        samples.ims.push_back(face);
        samples.labels.push_back(1);
    }

    for (auto &bg: sample_backgrounds(bgs, n_bgs)) {
        bg.normalize(1.0);
        samples.ims.push_back(bg);
        samples.labels.push_back(0);
    }

    return samples;
}

Stats
compute_stats(const images &ims) {
    double mean = 0;
    double std = 0;
    size_t total = 0;
    for (const auto &im: ims) {
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                mean += im.arr[y][x];
                total++;
            }
        }
    }
    mean /= (double) total;

    for (const auto &im: ims) {
        for (int y = 0; y < im.height; ++y) {
            for (int x = 0; x < im.width; ++x) {
                std += std::pow(im.arr[y][x] - mean, 2);
            }
        }
    }
    std /= (double) total;
    std = std::sqrt(std);

    return Stats{mean, std};
}

