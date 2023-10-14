#include <iostream>

#include "constants.h"
#include "utils.h"
#include "image.h"
#include "feature.h"


[[maybe_unused]] void test_load() {
    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    printf("Loaded: %zu face_paths, %zu backgrounds\n", face_paths.size(), bg_paths.size());

//    auto ims = sample_faces(face_paths, 7);
    auto ims = sample_backgrounds(bg_paths, 7, true);
    auto merged = merge_images(ims);

    cv::imshow("test", merged.toMat());
    cv::waitKey(0);
}

[[maybe_unused]] void integral_test() {
    auto fp = "../dataset/solvay-conference.jpg";

//    Img<uchar> im(IM_HEIGHT, IM_WIDTH);
    cv::Mat image = cv::imread(fp, cv::IMREAD_COLOR);
    Img<uchar> im(806, 1600);
    im.loadGrayScale(image);

    Img<float> im1 = im.cast<float>();
//    Img<float> integral = im1.toIntegral();
    // auto standard = integral.revertIntegral();
    im1.normalize(255);

    auto img = im1.toMat();

    cv::imshow("test", img);
    cv::waitKey(0);
}

[[maybe_unused]] void test_features() {
    Img<size_t> stan(5, 5);
    stan.arr[0] = new size_t[5]{5, 2, 3, 4, 1};
    stan.arr[1] = new size_t[5]{1, 5, 4, 2, 3};
    stan.arr[2] = new size_t[5]{2, 2, 1, 3, 4};
    stan.arr[3] = new size_t[5]{3, 5, 6, 4, 5};
    stan.arr[4] = new size_t[5]{4, 1, 3, 2, 6};


    auto fim = stan.cast<float>();
    auto integral = fim.toIntegral();
    std::cout << integral << std::endl;

    auto feature2h = Feature2h(3, 1, 2, 4);
    printf("diff: %d\n", feature2h.diff(integral));
    auto feature2v = Feature2v(3, 1, 2, 4);
    printf("diff: %d\n", feature2v.diff(integral));

    auto feature3h = Feature3h(3, 1, 2, 4);
    printf("diff: %d\n", feature3h.diff(integral));
    auto feature3v = Feature3v(3, 1, 2, 4);
    printf("diff: %d\n", feature3v.diff(integral));

    auto feature4 = Feature4(3, 1, 2, 4);
    printf("diff: %d\n", feature4.diff(integral));

    cv::imshow("test", integral.toMat());
    cv::waitKey(0);
}

[[maybe_unused]] void test_feature_iterators() {
    auto locs = possible_locations({4, 4}, 5);
    std::cout << "Locations: \n";
    for (const auto &loc: locs) {
        printf("(%d, %d)\n", loc.x, loc.y);
    }

    auto shapes = possible_shapes({4, 4}, 5);
    std::cout << "Shapes: \n";
    for (const auto &shape: shapes) {
        printf("(%d, %d)\n", shape.x, shape.y);
    }

    auto features = generate_features();
    print_features(features);
}

[[maybe_unused]] void test_normalize_images() {
    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    auto samples = sample_data(100, 100, face_paths, bg_paths);
    auto stats = compute_stats(samples.ims);
    printf("mean: %f, std: %f\n", stats.mean, stats.std);
    for (auto &im: samples.ims) {
        im.normalize((float) stats.mean, (float) stats.std);
    }
    stats = compute_stats(samples.ims);
    printf("mean: %f, std: %f\n", stats.mean, stats.std);
}

int main() {

}
