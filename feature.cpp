#include <iostream>
#include "feature.h"

std::vector<int>
possible_positions(int size, int window) {
    std::vector<int> positions;
    for (int i = 0; i < window - size + 1; ++i) {
        positions.push_back(i);
    }
    return positions;
}

std::vector<XY>
possible_locations(XY base_size, int window) {
    std::vector<XY> locations;
    for (int x: possible_positions(base_size.x, window)) {
        for (int y: possible_positions(base_size.y, window)) {
            locations.push_back({x, y});
        }
    }
    return locations;
}

std::vector<XY>
possible_shapes(XY base_size, int window) {
    std::vector<XY> shapes;
    for (int w = base_size.x; w <= window; w += base_size.x) {
        for (int h = base_size.y; h <= window; h += base_size.y) {
            shapes.push_back({w, h});
        }
    }
    return shapes;
}

Feature::Feature(size_t x, size_t y, size_t width, size_t height) {
    this->x = x;
    this->y = y;
    this->width = width;
    this->height = height;
}

void
Feature::print() const {
    std::cout << this->str() << std::endl;
}

std::string
Feature::str() const {
    std::string out = "Feature";
    out += this->name();
    out += "(x=";
    out += std::to_string(this->x);
    out += ", y=";
    out += std::to_string(this->y);
    out += ", w=";
    out += std::to_string(this->width);
    out += ", h=";
    out += std::to_string(this->height);
    out += ")";
    return out;
}

std::string
Feature::csv() const {
    std::string out = this->name();
    out += ",";
    out += std::to_string(this->x);
    out += ",";
    out += std::to_string(this->y);
    out += ",";
    out += std::to_string(this->width);
    out += ",";
    out += std::to_string(this->height);
    return out;
}

ImgFlt
Feature::diff(const ImgType &img) const {
    ImgFlt result = 0;
    auto [n, pts] = this->points();
    for (int i = 0; i < n; ++i) {
        auto pt = pts[i];
        if (pt.x >= img.width || pt.y >= img.height) {
            std::cout << "ERR[OUT_OF_BOUNDS] diff: %s" << str() << std::endl;
            continue;
        }
        result += (ImgFlt) pt.coef * img.arr[pt.y][pt.x];
    }
    return result;
}

std::tuple<int, const FeatPt *>
Feature2h::points() const {
    return {8, this->pts};
}

XY
Feature2h::baseSize() {
    return {2, 1};
}

const char *
Feature2h::name() const {
    return "2h";
}

Feature2h::Feature2h(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto hw = width / 2;
    this->pts[0] = {x, y, 1};
    this->pts[1] = {x + hw, y, -1};
    this->pts[2] = {x, y + height, -1};
    this->pts[3] = {x + hw, y + height, 1};

    this->pts[4] = {x + hw, y, -1};
    this->pts[5] = {x + width, y, 1};
    this->pts[6] = {x + hw, y + height, 1};
    this->pts[7] = {x + width, y + height, -1};
}

std::tuple<int, const FeatPt *>
Feature2v::points() const {
    return {8, this->pts};
}

XY
Feature2v::baseSize() {
    return {1, 2};
}

const char *
Feature2v::name() const {
    return "2v";
}

Feature2v::Feature2v(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto hh = height / 2;
    this->pts[0] = {x, y, -1};
    this->pts[1] = {x + width, y, 1};
    this->pts[2] = {x, y + hh, 1};
    this->pts[3] = {x + width, y + hh, -1};

    this->pts[4] = {x, y + hh, 1};
    this->pts[5] = {x + width, y + hh, -1};
    this->pts[6] = {x, y + height, -1};
    this->pts[7] = {x + width, y + height, 1};
}

std::tuple<int, const FeatPt *>
Feature3h::points() const {
    return {12, this->pts};
}

XY
Feature3h::baseSize() {
    return {3, 1};
}

const char *
Feature3h::name() const {
    return "3h";
}

Feature3h::Feature3h(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto tw = width / 3;
    this->pts[0] = {x, y, -1};
    this->pts[1] = {x + tw, y, 1};
    this->pts[2] = {x, y + height, 1};
    this->pts[3] = {x + tw, y + height, -1};

    this->pts[4] = {x + tw, y, 1};
    this->pts[5] = {x + 2 * tw, y, -1};
    this->pts[6] = {x + tw, y + height, -1};
    this->pts[7] = {x + 2 * tw, y + height, 1};

    this->pts[8] = {x + 2 * tw, y, -1};
    this->pts[9] = {x + width, y, 1};
    this->pts[10] = {x + 2 * tw, y + height, 1};
    this->pts[11] = {x + width, y + height, -1};
}

std::tuple<int, const FeatPt *>
Feature3v::points() const {
    return {12, this->pts};
}

XY
Feature3v::baseSize() {
    return {1, 3};
}

const char *
Feature3v::name() const {
    return "3v";
}

Feature3v::Feature3v(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto th = height / 3;
    this->pts[0] = {x, y, -1};
    this->pts[1] = {x + width, y, 1};
    this->pts[2] = {x, y + th, 1};
    this->pts[3] = {x + width, y + th, -1};

    this->pts[4] = {x, y + th, 1};
    this->pts[5] = {x + width, y + th, -1};
    this->pts[6] = {x, y + 2 * th, -1};
    this->pts[7] = {x + width, y + 2 * th, 1};

    this->pts[8] = {x, y + 2 * th, -1};
    this->pts[9] = {x + width, y + 2 * th, 1};
    this->pts[10] = {x, y + height, 1};
    this->pts[11] = {x + width, y + height, -1};
}

std::tuple<int, const FeatPt *>
Feature4::points() const {
    return {16, this->pts};
}

XY
Feature4::baseSize() {
    return {2, 2};
}

const char *
Feature4::name() const {
    return "4r";
}

Feature4::Feature4(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto hw = width / 2;
    auto hh = height / 2;

    // Upper row
    this->pts[0] = {x, y, 1};
    this->pts[1] = {x + hw, y, -1};
    this->pts[2] = {x, y + hh, -1};
    this->pts[3] = {x + hw, y + hh, 1};

    this->pts[4] = {x + hw, y, -1};
    this->pts[5] = {x + width, y, 1};
    this->pts[6] = {x + hw, y + hh, 1};
    this->pts[7] = {x + width, y + hh, -1};

    // Lower row
    this->pts[8] = {x, y + hh, -1};
    this->pts[9] = {x + hw, y + hh, 1};
    this->pts[10] = {x, y + height, 1};
    this->pts[11] = {x + hw, y + height, -1};

    this->pts[12] = {x + hw, y + hh, 1};
    this->pts[13] = {x + width, y + hh, -1};
    this->pts[14] = {x + hw, y + height, -1};
    this->pts[15] = {x + width, y + height, 1};
}

Features
generate_features() {
    std::vector<Feature2h> features2h;
    for (XY shape: possible_shapes(Feature2h::baseSize(), FEATURE_SIZE)) {
        for (XY loc: possible_locations({shape.x, shape.y}, FEATURE_SIZE)) {
            features2h.emplace_back(loc.x, loc.y, shape.x, shape.y);
        }
    }
    std::vector<Feature2v> features2v;
    for (XY shape: possible_shapes(Feature2v::baseSize(), FEATURE_SIZE)) {
        for (XY loc: possible_locations({shape.x, shape.y}, FEATURE_SIZE)) {
            features2v.emplace_back(loc.x, loc.y, shape.x, shape.y);
        }
    }
    std::vector<Feature3h> features3h;
    for (XY shape: possible_shapes(Feature3h::baseSize(), FEATURE_SIZE)) {
        for (XY loc: possible_locations({shape.x, shape.y}, FEATURE_SIZE)) {
            features3h.emplace_back(loc.x, loc.y, shape.x, shape.y);
        }
    }
    std::vector<Feature3v> features3v;
    for (XY shape: possible_shapes(Feature3v::baseSize(), FEATURE_SIZE)) {
        for (XY loc: possible_locations({shape.x, shape.y}, FEATURE_SIZE)) {
            features3v.emplace_back(loc.x, loc.y, shape.x, shape.y);
        }
    }
    std::vector<Feature4> features4;
    for (XY shape: possible_shapes(Feature4::baseSize(), FEATURE_SIZE)) {
        for (XY loc: possible_locations({shape.x, shape.y}, FEATURE_SIZE)) {
            features4.emplace_back(loc.x, loc.y, shape.x, shape.y);
        }
    }

    return {features2h, features2v, features3h, features3v, features4};
}

void
print_features(const Features &features) {
    auto total = features.f2h.size() + features.f2v.size() + features.f3h.size() + features.f3v.size() +
                 features.f4.size();
    printf("Features (%zu):\n", total);
    printf("\t2h: %zu\n", features.f2h.size());
    printf("\t2v: %zu\n", features.f2v.size());
    printf("\t3h: %zu\n", features.f3h.size());
    printf("\t4:  %zu\n", features.f4.size());
    printf("\t3v: %zu\n", features.f3v.size());

    // get actual size of features in bytes
    size_t total_size = 0;
    total_size += features.f2h.size() * sizeof features.f2h[0];
    total_size += features.f2v.size() * sizeof features.f2v[0];
    total_size += features.f3h.size() * sizeof features.f3h[0];
    total_size += features.f3v.size() * sizeof features.f3v[0];
    total_size += features.f4.size() * sizeof features.f4[0];
    auto mb = (double) total_size / 1024.0 / 1024.0;
    printf("Features use %.2f MB\n", mb);
}

std::vector<shdptr<Feature>>
feature_vec(const Features &features) {
    std::vector<shdptr<Feature>> feats;
    for (auto &f: features.f2h) feats.push_back(mkshd<Feature2h>(f));
    for (auto &f: features.f2v) feats.push_back(mkshd<Feature2v>(f));
    for (auto &f: features.f3h) feats.push_back(mkshd<Feature3h>(f));
    for (auto &f: features.f3v) feats.push_back(mkshd<Feature3v>(f));
    for (auto &f: features.f4) feats.push_back(mkshd<Feature4>(f));
    return feats;
}