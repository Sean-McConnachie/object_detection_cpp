#pragma once

#include <cstddef>
#include "image.h"

typedef struct {
    size_t x;
    size_t y;
    int coef;
} FeatPt;

typedef struct {
    int x;
    int y;
} XY;

std::vector<int>
possible_positions(int size, int window);

std::vector<XY>
possible_locations(XY base_size, int window);

std::vector<XY>
possible_shapes(XY base_size, int window);

class Feature {
private:
public:
    size_t x;
    size_t y;
    size_t width;
    size_t height;

    Feature(size_t x, size_t y, size_t width, size_t height);

    Feature(const Feature &other) = default;

    void print() const;

    [[nodiscard]] std::string str() const;

    [[nodiscard]] std::string csv() const;

    virtual std::tuple<int, const FeatPt *> points() const = 0;

    [[nodiscard]] virtual const char *name() const = 0;

    [[nodiscard]] ImgFlt diff(const ImgType &img) const;
};

class Feature2h : public Feature {
private:
public:
    FeatPt pts[8]{};

    Feature2h(size_t x, size_t y, size_t width, size_t height);

    [[nodiscard]] std::tuple<int, const FeatPt *> points() const override;

    [[nodiscard]] static XY baseSize();

    [[nodiscard]] const char *name() const override;
};

class Feature2v : public Feature {
private:
public:
    FeatPt pts[8]{};

    Feature2v(size_t x, size_t y, size_t width, size_t height);

    [[nodiscard]] std::tuple<int, const FeatPt *> points() const override;

    [[nodiscard]] static XY baseSize();

    [[nodiscard]] const char *name() const override;
};

class Feature3h : public Feature {
private:
public:
    FeatPt pts[12]{};

    Feature3h(size_t x, size_t y, size_t width, size_t height);

    [[nodiscard]] std::tuple<int, const FeatPt *> points() const override;

    [[nodiscard]] static XY baseSize();

    [[nodiscard]] const char *name() const override;

};

class Feature3v : public Feature {
private:
public:
    FeatPt pts[12]{};

    Feature3v(size_t x, size_t y, size_t width, size_t height);

    [[nodiscard]] std::tuple<int, const FeatPt *> points() const override;

    [[nodiscard]] static XY baseSize();

    [[nodiscard]] const char *name() const override;
};

class Feature4 : public Feature {
private:
public:
    FeatPt pts[16]{};

    Feature4(size_t x, size_t y, size_t width, size_t height);

    [[nodiscard]] std::tuple<int, const FeatPt *> points() const override;

    [[nodiscard]] static XY baseSize();

    [[nodiscard]] const char *name() const override;
};

typedef struct {
    std::vector<Feature2h> f2h;
    std::vector<Feature2v> f2v;
    std::vector<Feature3h> f3h;
    std::vector<Feature3v> f3v;
    std::vector<Feature4> f4;
} Features;

Features generate_features();

void print_features(const Features &features);

std::vector<shdptr<Feature>> feature_vec(const Features &features);