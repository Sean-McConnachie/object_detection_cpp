#pragma once

#include "constants.h"
#include "learner.h"

typedef std::vector<std::tuple<XY, XY>> boxes;

class Runtime {
private:
    vec<vec<classifiervec>> cascadeAtScales;
public:
    Runtime(vec<classifiervec> cascade);

    ~Runtime() = default;

    boxes run(ImgType &img) const;

    static void drawBoxes(cv::Mat &img, const boxes &b);
};