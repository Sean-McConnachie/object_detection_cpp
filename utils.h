#pragma once

#include "constants.h"
#include <opencv2/opencv.hpp>


template<typename T>
class Img {
private:
public:
    size_t height{};
    size_t width{};
    T **arr;

    Img(size_t height, size_t width);

    ~Img();

    Result loadGrayScale(const std::string &path);

    Img<size_t> toIntegral();

    cv::Mat toMat();

    void normalizeTo(size_t max);

    [[maybe_unused]] Img<uchar> revertIntegral();
};

template
class Img<uchar>;

template
class Img<size_t>;

