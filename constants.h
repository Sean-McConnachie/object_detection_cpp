#pragma once

#include <memory>
#include <vector>

#define FEATURE_SIZE 24
#define IM_WIDTH 384
#define IM_HEIGHT 288
#define FP_FACES_DIR "../dataset/faces/"
#define FP_BGS_DIR "../dataset/backgrounds/"
#define FACES_CROP_TOP 50
#define STATUS_EVERY 10000
#define SCALE_FACTOR 1.25

typedef unsigned char uchar;

template<typename T>
inline
std::shared_ptr<T> mkshd(T x) {
    return std::make_shared<T>(x);
}

template<typename T>
inline
std::unique_ptr<T> mkunq(T x) {
    return std::make_unique<T>(x);
}

template<typename T>
using shdptr = std::shared_ptr<T>;

template<typename T>
using unqptr = std::unique_ptr<T>;

template<typename T>
using vec = std::vector<T>;