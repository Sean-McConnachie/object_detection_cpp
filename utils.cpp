#include "utils.h"

template<typename T>
Img<T>::Img(size_t height, size_t width) {
    this->height = height;
    this->width = width;
    this->arr = new T *[height];
    for (size_t i = 0; i < height; i++) {
        this->arr[i] = new T[width];
    }
}

template<typename T>
Img<T>::~Img() {
    for (size_t i = 0; i < this->height; i++) {
        delete[] this->arr[i];
    }
    delete[] this->arr;
}

template<typename T>
Img<size_t> Img<T>::toIntegral() {
    Img<size_t> integral(this->height + 1, this->width + 1);
    for (size_t x = 0; x < integral.height; ++x) {
        integral.arr[0][x] = 0;
    }
    for (size_t y = 0; y < integral.height; ++y) {
        integral.arr[y][0] = 0;
    }
    for (size_t y = 0; y < this->height; ++y) {
        for (size_t x = 0; x < this->width; ++x) {
            integral.arr[y + 1][x + 1] =
                    this->arr[y][x]
                    + integral.arr[y][x + 1]
                    + integral.arr[y + 1][x]
                    - integral.arr[y][x];
        }
    }
    return integral;
}

template<typename T>
Result
Img<T>::loadGrayScale(const std::string &path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "Could not read the image: " << path << std::endl;
        return FAILURE;
    }
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(this->width, this->height));
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            this->arr[y][x] = static_cast<T>(resized.at<uchar>((int) y, (int) x));
        }
    }
    return SUCCESS;
}

template<typename T>
cv::Mat
Img<T>::toMat() {
    cv::Mat mat(this->height, this->width, CV_8UC1);//CV_8UC1
    for (int y = 0; y < this->height; y++) {
        for (int x = 0; x < this->width; x++) {
            mat.at<uchar>(y, x) = this->arr[y][x];
        }
    }
    return mat;
}

template<typename T>
void
Img<T>::normalizeTo(size_t max) {
    size_t max_val = 0;
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            if (this->arr[y][x] > max_val) {
                max_val = this->arr[y][x];
            }
        }
    }

    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            this->arr[y][x] = (this->arr[y][x] * max) / max_val;
        }
    }
}

template<typename T>
[[maybe_unused]] Img<uchar> Img<T>::revertIntegral() {
    Img<uchar> integral(this->height - 1, this->width - 1);
    for (size_t y = 0; y < this->height - 1; ++y) {
        for (size_t x = 0; x < this->width - 1; ++x) {
            integral.arr[y][x] = this->arr[y + 1][x + 1] - this->arr[y][x + 1] - this->arr[y + 1][x] + this->arr[y][x];
        }
    }
    return integral;
}
