#include "utils.h"

std::vector<std::string> list_dir(const std::string &path) {
    std::vector<std::string> paths;
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        paths.push_back(entry.path());
    }
    return paths;
}

Img<float>
open_face(const std::string &path) {
    Img<float> im(FEATURE_SIZE, FEATURE_SIZE);

    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    cv::Rect roi(0, FACES_CROP_TOP, image.cols, image.rows - FACES_CROP_TOP);
    cv::Mat croppedImage = image(roi);
    int min = std::min(croppedImage.cols, croppedImage.rows);
    cv::Rect fitRect((croppedImage.cols - min) / 2, (croppedImage.rows - min) / 2, min, min);
    cv::Mat fitImage = croppedImage(fitRect);

    im.loadGrayScale(fitImage);

    return im;
}

Img<float>
merge_images(const std::vector<Img<float>> &images) {
    int max_height = 0;
    int total_width = 0;
    for (const auto &image: images) {
        max_height = std::max(max_height, static_cast<int>(image.height));
        total_width += (int) image.width;
    }

    Img<float> merged(max_height, total_width);
    for (int i = 0; i < images.size(); ++i) {
        for (int y = 0; y < images[i].height; ++y) {
            for (int x = 0; x < images[i].width; ++x) {
                merged.arr[y][x + i * images[i].width] = images[i].arr[y][x];
            }
        }
    }

    return merged;
}

std::vector<Img<float>>
sample_ims(const paths &ims, size_t n) {
    std::vector<Img<float>> result;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, ims.size() - 1);

    for (size_t i = 0; i < n; ++i) {
        result.push_back(open_face(ims[distrib(gen)]));
    }

    return result;
}