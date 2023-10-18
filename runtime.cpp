#include "runtime.h"

shdptr<Feature> copyFeature(const std::string &name, int x, int y, int width, int height) {
    shdptr<Feature> f;
    if (name == "2h") {
        f = std::make_shared<Feature2h>(x, y, width, height);
    } else if (name == "2v") {
        f = std::make_shared<Feature2v>(x, y, width, height);
    } else if (name == "3h") {
        f = std::make_shared<Feature3h>(x, y, width, height);
    } else if (name == "3v") {
        f = std::make_shared<Feature3v>(x, y, width, height);
    } else if (name == "4r") {
        f = std::make_shared<Feature4>(x, y, width, height);
    } else {
        throw std::runtime_error("Unknown feature name: " + std::string(name));
    }
    return f;
}

int
scaleUp(int val, flt factor) {
    return (int) ((flt) val * factor);
}

Runtime::Runtime(vec<classifiervec> cascade) {
    int scale_i = 0;
    flt max_y = (flt) FEATURE_SIZE;
    flt max_x = (flt) FEATURE_SIZE;
    flt scale = 1.0;
    size_t total = 0;
    while (max_y < IM_HEIGHT && max_x < IM_WIDTH) {
        cascadeAtScales.emplace_back();
        cascadeAtScales[scale_i].reserve(cascade.size());
        int layer_i = 0;
        for (const auto &layer: cascade) {
            cascadeAtScales[scale_i].emplace_back();
            cascadeAtScales[scale_i][layer_i].reserve(layer.size());
            for (const auto &c: layer) {
                auto f = copyFeature(c.feat->name(), scaleUp(c.feat->x, scale), scaleUp(c.feat->y, scale),
                                     scaleUp(c.feat->width, scale), scaleUp(c.feat->height, scale));
                WeakClassifier wc{c.threshold, c.polarity, c.alpha, f};
                cascadeAtScales[scale_i][layer_i].push_back(wc);
            }
            layer_i++;
        }
        scale_i++;
        max_y = FEATURE_SIZE * scale;
        max_x = FEATURE_SIZE * scale;
        scale += SCALE_FACTOR;
        break;
    }
}

boxes
Runtime::run(ImgType &img) const {
    // (pos, size)
    std::vector<std::tuple<XY, XY>> locs;

    int scale_i = 0;
    flt max_y = (flt) FEATURE_SIZE;
    flt max_x = (flt) FEATURE_SIZE;
    flt scale = 1.0;
    size_t total = 0;

    while (max_y < img.height && max_x < img.width) {
        ImgType cropped((int) max_y, (int) max_x);
        auto half_y = max_y / 2;
        auto half_x = max_x / 2;
        for (int y = (int) half_y; y + max_y < img.height; y++) {
            for (int x = (int) half_x; x + max_x < img.width; x++) {
                for (const auto &layer: cascadeAtScales[scale_i]) {
                    cropped = img.cropForIntegral(x - half_x, y - half_y, (int) max_x, (int) max_y);
                    StrongClassifierResult h = Learner::strongClassifier(cropped, layer);
                    total++;
                    if (!(h.weightedSum >= h.alphaSum * 0.5)) {
                        goto exit;
                    }
                    // TODO: Threshold for strongClassifier save
//                    int lbl = h.label();
//                    if (lbl == 1) {
//                        locs.push_back({{x,           y},
//                                        {(int) max_x, (int) max_y}});
//                    }
                }
                locs.push_back({{x - (int) half_x, y - (int) half_y},
                                {(int) max_x,      (int) max_y}});
                exit:;
            }
        }

        scale_i++;
        max_y = FEATURE_SIZE * scale;
        max_x = FEATURE_SIZE * scale;
        scale += SCALE_FACTOR;
        break;
    }

    printf("Found %zu faces out of %zu windows.\n", locs.size(), total);
    return locs;
}

void
Runtime::drawBoxes(cv::Mat &img, const boxes &b) {
    for (const auto &box: b) {
        auto [x, y] = std::get<0>(box);
        auto [w, h] = std::get<1>(box);
        cv::Point p1(x, y);
        cv::Point p2(x + w, y + h);
        cv::rectangle(img, p1, p2, {0, 255, 0}, 1);
    }
}