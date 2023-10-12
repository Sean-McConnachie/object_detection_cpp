#include <iostream>

#include "utils.h"

int main() {
//    auto fp = "../dataset/faces/1905_Ohio_Cleveland_Central_0-1.png";
    auto fp = "../dataset/solvay-conference.jpg";

    Img<uchar> im(IM_HEIGHT, IM_WIDTH);
    im.loadGrayScale(fp);

    Img<size_t> integral = im.toIntegral();

    integral.normalizeTo(255);
    // auto standard = integral.revertIntegral();

    auto img = integral.toMat();

    cv::imshow("test", img);
    cv::waitKey(0);
}
