#pragma once

#include "constants.h"
#include <opencv2/opencv.hpp>

void gamma(cv::Mat &img, double gamma);
cv::Mat gleam(cv::Mat &img);

template<typename T>
class Img {
private:
public:
    int height{};
    int width{};
    T **arr;

    Img(int height, int width);

    /**
     * @brief Copy constructor
     * @param img
     */
    Img(const Img<T> &img);

    ~Img();

    /**
     * @brief Apply gamma and gleam.
     * @param image
     * @return
     */
    void loadGrayScale(cv::Mat image);

    Img<T> toIntegral() const;

    template<typename U>
    Img<U> cast();

    /**
     * @brief Convert to cv::Mat. Assumes image is normalized to [0, 255].
     * @return cv::Mat
     */
    cv::Mat toMat();

    void normalize(T max);

    void normalize(T mean, T std);

    [[maybe_unused]] Img<uchar> revertIntegral();

    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Img<U> &img);

    void print() { std::cout << *this << std::endl; }

    Img<T> resize(int h, int w);
};

template<typename U>
std::ostream &operator<<(std::ostream &os, const Img<U> &img) {
    for (size_t y = 0; y < img.height; y++) {
        for (size_t x = 0; x < img.width; x++) {
            os << img.arr[y][x] << '\t';
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
template<typename U>
Img<U> Img<T>::cast() {
    Img<U> result(height, width);

    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            result.arr[i][j] = static_cast<U>(arr[i][j]);
        }
    }

    return result;
}

template
class Img<uchar>;

template
class Img<size_t>;

template
class Img<float>;

template
class Img<double>;
