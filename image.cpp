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
    this->arr.reserve(height);
    for (size_t i = 0; i < height; ++i) {
        std::vector<T> row(width);
        this->arr.push_back(row);
    }
}

template<typename T>
void
Img<T>::swap(Img<T> &other) {
    std::swap(height, other.height);
    std::swap(width, other.width);
    std::swap(arr, other.arr);
}

template<typename T>
Img<T>::~Img() {
//    for (auto &row: this->arr) {
//        row.clear();
//    }
//    this->arr.clear();
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
Img<T>::normalize() {
    double mean = 0;
    double std = 0;
    size_t total = 0;
    for (int y = 0; y < this->height; ++y) {
        for (int x = 0; x < this->width; ++x) {
            mean += this->arr[y][x];
            total++;
        }
    }
    mean /= (double) total;

    for (int y = 0; y < this->height; ++y) {
        for (int x = 0; x < this->width; ++x) {
            std += std::pow(this->arr[y][x] - mean, 2);
        }
    }
    std /= (double) total;
    std = std::sqrt(std);

    normalize(mean, std);
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
void
Img<T>::rangeTo(T max) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            this->arr[y][x] = this->arr[y][x] / max;
        }
    }
}

template<typename T>
[[maybe_unused]] Img<T> Img<T>::revertIntegral() {
    Img<T> integral(this->height - 1, this->width - 1);
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

template<typename T>
Img<T> Img<T>::cropForIntegral(int x, int y, int w, int h) {
    Img<T> cropped(h + 1, w + 1);
    for (int row = 0; row < h + 1; ++row) {
        for (int col = 0; col < w + 1; ++col) {
            if (row + y >= this->height || col + x >= this->width) {
                printf("ERR[OUT_OF_BOUNDS] Img::Crop (%d, %d)\n", row + y, col + x);
                continue;
            }
            cropped.arr[row][col] = this->arr[row + y][col + x];
        }
    }
    return cropped;
}

Scale
scaled_size(Scale max, Scale size) {
    Scale scaled;
    if (size.width > size.height) {
        scaled.width = max.width;
        scaled.height = (int) ((ImgFlt) max.width * ((ImgFlt) size.height / (ImgFlt) size.width));
    } else {
        scaled.height = max.height;
        scaled.width = (int) ((ImgFlt) max.height * ((ImgFlt) size.width / (ImgFlt) size.height));
    }
    return scaled;
}