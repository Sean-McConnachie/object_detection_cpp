#include "image.h"

void
gamma(cv::Mat &img, double gleam) {
    gleam = 1.0 / gleam;
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            auto pix = img.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                img.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(
                        std::pow(pix[c] / 255.0, gleam) * 255.0);
            }
        }
    }
}

cv::Mat
gleam(cv::Mat &img) {
    gamma(img, 2.2);
    cv::Mat gleamed(img.size(), CV_8UC1);
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            gleamed.at<uchar>(y, x) = 0;
            auto pix = img.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                gleamed.at<uchar>(y, x) += cv::saturate_cast<uchar>(pix[c] / 3.0);
            }
        }
    }
    return gleamed;
}

template<typename T>
Img<T>::Img(int height, int width) {
    this->height = height;
    this->width = width;
    this->arr = new T *[height];
    for (size_t i = 0; i < height; i++) {
        this->arr[i] = new T[width];
    }
}

template<typename T>
Img<T>::Img(const Img<T> &img) {
    this->height = img.height;
    this->width = img.width;
    this->arr = new T *[height];
    for (size_t i = 0; i < height; i++) {
        this->arr[i] = new T[width];
        for (size_t j = 0; j < width; j++) {
            this->arr[i][j] = img.arr[i][j];
        }
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
Img<T> Img<T>::toIntegral() const {
    Img<T> integral(this->height + 1, this->width + 1);
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
void
Img<T>::loadGrayScale(cv::Mat image) {
    cv::Mat gleamed = gleam(image);

    cv::Mat resized;
    cv::resize(gleamed, resized, cv::Size(this->width, this->height));

    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            this->arr[y][x] = static_cast<T>(resized.at<uchar>((int) y, (int) x));
        }
    }
}

template<typename T>
cv::Mat
Img<T>::toMat() {
    cv::Mat mat(this->height, this->width, CV_8UC1);
    for (int y = 0; y < this->height; y++) {
        for (int x = 0; x < this->width; x++) {
            mat.at<uchar>(y, x) = this->arr[y][x];
        }
    }
    return mat;
}

template<typename T>
void
Img<T>::normalize(T max) {
    T max_val = 0;
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
void
Img<T>::normalize(T mean, T std) {
    for (size_t y = 0; y < this->height; y++) {
        for (size_t x = 0; x < this->width; x++) {
            this->arr[y][x] = (this->arr[y][x] - mean) / std;
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

template<typename T>
Img<T> Img<T>::resize(int h, int w) {
    Img<T> resized(h, w);
    cv::Mat mat = this->toMat();
    cv::resize(mat, mat, cv::Size(w, h));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            resized.arr[y][x] = mat.at<uchar>(y, x);
        }
    }
    return resized;
}
