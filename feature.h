#pragma once

#include <cstddef>

class Feature {
private:
public:
    size_t x;
    size_t y;
    size_t width;
    size_t height;

    Feature(size_t x, size_t y, size_t width, size_t height);
};

typedef struct {
    size_t x;
    size_t y;
    int coef;
} FeatPt;

class Feature2h : public Feature {
private:
public:
    FeatPt pts[8];

    Feature2h(size_t x, size_t y, size_t width, size_t height);
};

class Feature2v : public Feature {
private:
public:
    FeatPt pts[8];

    Feature2v(size_t x, size_t y, size_t width, size_t height);
};

class Feature3h : public Feature {
private:
public:
    FeatPt pts[12];

    Feature3h(size_t x, size_t y, size_t width, size_t height);
};

class Feature3v : public Feature {
private:
public:
    FeatPt pts[12];

    Feature3v(size_t x, size_t y, size_t width, size_t height);
};

class Feature4 : public Feature {
private:
public:
    FeatPt pts[16];

    Feature4(size_t x, size_t y, size_t width, size_t height);
};