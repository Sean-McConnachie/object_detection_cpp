#include <iostream>

#include "constants.h"
#include "utils.h"
#include "image.h"

int main() {
    auto faces = list_dir(FP_FACES_DIR);
    auto bgs = list_dir(FP_BGS_DIR);
    printf("Loaded: %zu faces, %zu backgrounds\n", faces.size(), bgs.size());

//    auto f = open_face(faces[123]);
//    auto f1 = open_face(faces[124]);
    auto ims = sample_ims(faces, 50);

    auto merged = merge_images(ims);

    cv::imshow("test", merged.toMat());
    cv::waitKey(0);


    /*
    Img<size_t> stan(5, 5);
    stan.arr[0] = new size_t[5]{5, 2, 3, 4, 1};
    stan.arr[1] = new size_t[5]{1, 5, 4, 2, 3};
    stan.arr[2] = new size_t[5]{2, 2, 1, 3, 4};
    stan.arr[3] = new size_t[5]{3, 5, 6, 4, 5};
    stan.arr[4] = new size_t[5]{4, 1, 3, 2, 6};

    auto inte = stan.toIntegral();


    auto im = stan.toMat();
    cv::imshow("test", im);
    cv::waitKey(0);
     */

    /*
//    auto fp = "../dataset/faces/1905_Ohio_Cleveland_Central_0-1.png";
    auto fp = "../dataset/solvay-conference.jpg";

//    Img<uchar> im(IM_HEIGHT, IM_WIDTH);
    cv::Mat image = cv::imread(fp, cv::IMREAD_COLOR);
    Img<uchar> im(806, 1600);
    im.loadGrayScale(image);

    Img<float> im1 = im.cast<float>();

//    Img<float> integral = im1.toIntegral();

    im1.normalizeTo(255);

    // auto standard = integral.revertIntegral();

    auto img = im1.toMat();

    cv::imshow("test", img);
    cv::waitKey(0);
    */
}
