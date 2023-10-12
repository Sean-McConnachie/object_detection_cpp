#include "feature.h"

Feature::Feature(size_t x, size_t y, size_t width, size_t height) {
    this->x = x;
    this->y = y;
    this->width = width;
    this->height = height;
}

Feature2h::Feature2h(size_t x, size_t y, size_t width, size_t height) : Feature(x, y, width, height) {
    auto hw = width / 2;
    this->pts[0] = {x, y, 1};
    this->pts[1] = {x + hw, y, -1};
    this->pts[2] = {x, y + height, -1};
    this->pts[3] = {x + hw, y + height, -1};

    this->pts[4] = {x + hw, y, -1};
    this->pts[5] = {x + width, y, 1};
    this->pts[6] = {x + hw, y + height, 1};
    this->pts[7] = {x + width, y + height, -1};
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