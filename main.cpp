#include <iostream>
#include <fstream>

#include "constants.h"
#include "utils.h"
#include "image.h"
#include "feature.h"
#include "learner.h"

[[maybe_unused]] void test_load() {
    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    printf("Loaded: %zu face_paths, %zu backgrounds\n", face_paths.size(), bg_paths.size());

//    auto images = sample_faces(face_paths, 7);
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

    ImgType im1 = im.cast<ImgFlt>();
    ImgType integral = im1.toIntegral();


    im1.normalize(255);

    auto standard = integral.revertIntegral();

    cv::imshow("test", standard.toMat());
    cv::waitKey(0);
}

[[maybe_unused]] void test_features() {
    Img<size_t> stan(5, 5);
    stan.arr[0] = {5, 2, 3, 4, 1};
    stan.arr[1] = {1, 5, 4, 2, 3};
    stan.arr[2] = {2, 2, 1, 3, 4};
    stan.arr[3] = {3, 5, 6, 4, 5};
    stan.arr[4] = {4, 1, 3, 2, 6};


    auto fim = stan.cast<ImgFlt>();
    auto integral = fim.toIntegral();
    std::cout << integral << std::endl;

    auto feature2h = Feature2h(3, 1, 2, 4);
    printf("diff: %f\n", feature2h.diff(integral));
    auto feature2v = Feature2v(3, 1, 2, 4);
    printf("diff: %f\n", feature2v.diff(integral));

    auto feature3h = Feature3h(3, 1, 2, 4);
    printf("diff: %f\n", feature3h.diff(integral));
    auto feature3v = Feature3v(3, 1, 2, 4);
    printf("diff: %f\n", feature3v.diff(integral));

    auto feature4 = Feature4(3, 1, 2, 4);
    printf("diff: %f\n", feature4.diff(integral));

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
        im.normalize((ImgFlt) stats.mean, (ImgFlt) stats.std);
    }
    stats = compute_stats(samples.ims);
    printf("mean: %f, std: %f\n", stats.mean, stats.std);
}

[[maybe_unused]] void test_feature_extraction() {
    // ENSURE WINDOW_SIZE is 15
    const int FACE_COUNT = 1000;
    const int BG_COUNT = 1000;
    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    auto samples = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
    auto stats = compute_stats(samples.ims);

    Samples norms;
    for (auto &im: samples.ims) {
        im.normalize((ImgFlt) stats.mean, (ImgFlt) stats.std);
        auto integral = im.toIntegral();
        norms.ims.push_back(integral);
    }

    auto test_feature = Feature2v(2, 3, 10, 4);
    std::vector<ImgFlt> zs;

    for (auto &im: norms.ims) {
        zs.push_back(test_feature.diff(im));
    }

    double face_mean = 0;
    double bg_mean = 0;
    for (int i = 0; i < zs.size(); ++i) {
        if (samples.labels[i] == 1) {
            face_mean += zs[i];
        } else {
            bg_mean += zs[i];
        }
    }
    face_mean /= FACE_COUNT;
    bg_mean /= BG_COUNT;

    double face_std = 0;
    double bg_std = 0;
    for (int i = 0; i < zs.size(); ++i) {
        if (samples.labels[i] == 1) {
            face_std += std::pow(zs[i] - face_mean, 2);
        } else {
            bg_std += std::pow(zs[i] - bg_mean, 2);
        }
    }
    face_std /= FACE_COUNT;
    bg_std /= BG_COUNT;
    face_std = std::sqrt(face_std);
    bg_std = std::sqrt(bg_std);

    printf("face_mean: %f, face_std: %f\n", face_mean, face_std);
    printf("bg_mean: %f, bg_std: %f\n", bg_mean, bg_std);

    // write zs to file
    std::ofstream myfile;
    myfile.open("../pywork/o.txt");
    for (int i = 0; i < zs.size(); ++i) {
        myfile << samples.labels[i] << " " << zs[i] << "\n";
    }
}

void draw_box(cv::Mat &img, int x, int y, int w, int h) {
    cv::Point p1(x, y);
    cv::Point p2(x + w, y + h);
    cv::rectangle(img, p1, p2, {0, 255, 0}, 1);
}

void test_classifier(const std::vector<std::vector<WeakClassifier>> &weak_classifiers, const std::string &impath) {
    cv::Mat im = cv::imread(impath, cv::IMREAD_COLOR);
    Scale scaled = scaled_size({IM_WIDTH, IM_HEIGHT}, {im.cols, im.rows});
    cv::Mat resized;
    cv::resize(im, resized, {scaled.width, scaled.height});

    ImgType img(scaled.height, scaled.width);
    img.loadGrayScale(resized);
    img.normalize();
    ImgType integral = img.toIntegral();

    const int HALF_WINDOW = FEATURE_SIZE / 2;
    std::vector<std::vector<XY>> locations;
    for (int i = 0; i < weak_classifiers.size(); i++) locations.emplace_back();
    for (int row = HALF_WINDOW + 1; row < scaled.height - HALF_WINDOW; ++row) {
        for (int col = HALF_WINDOW + 1; col < scaled.width - HALF_WINDOW; ++col) {
            auto top = row - HALF_WINDOW - 1;
            auto left = col - HALF_WINDOW - 1;
            ImgType cropped = integral.cropForIntegral(left, top, FEATURE_SIZE, FEATURE_SIZE);
            int i = 0;
            for (const auto &classifier: weak_classifiers) {
                int probable_face = Learner::strongClassifier(cropped, classifier);
                if (probable_face == 0) goto skip;
                locations[i++].push_back({left, top});
            }
            draw_box(resized, left, top, FEATURE_SIZE, FEATURE_SIZE);
            skip:;
        }
    }
    for (int i = 0; i < locations.size(); i++) {
        printf("classifier %d: %zu\n", i, locations[i].size());
    }
    cv::imshow("test", resized);
    cv::waitKey(0);
}

int main() {
    const char *CLASSIFIER_PATH[] = {
            "../classifiers/15px_4layer_1000faces.csv",
            "../classifiers/15px_10layer_1000faces.csv",
            "../classifiers/15px_25layer_1000faces.csv"
    };

//    const char *CLASSIFIER_PATH[] = {
//            "../classifiers/24px_4layer_1000faces.csv",
//            "../classifiers/24px_38layer_1000faces.csv"
//    };

    std::vector<std::vector<WeakClassifier >> classifiers;
    for (const auto &p: CLASSIFIER_PATH) {
        classifiers.push_back(load_weak_classifiers(p));
        print_weak_classifiers(classifiers.back());
    }

//    for (int i = 0; i < 10; ++i) {
//        classifiers[0].pop_back();
//    }

    test_classifier(classifiers, "../dataset/legend0.jpg");

//    const int NUM_CASCADE_LAYERS = 25;
//    const char *CLASSIFIER_PATH = "../classifiers/15px_25layer_1000faces.csv";
//    const int FACE_COUNT = 2000;
//    const int BG_COUNT = 2000;
//
//    auto face_paths = list_dir(FP_FACES_DIR);
//    auto bg_paths = list_dir(FP_BGS_DIR);
//    auto samples = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
//    auto stats = compute_stats(samples.ims);
//    samples = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
//    printf("mean: %f, std: %f\n", stats.mean, stats.std);
//
//    auto features = generate_features();
//    print_features(features);
//    Learner learner(samples.ims, samples.labels, features, stats);
//
//    auto weak_classifiers = learner.train(NUM_CASCADE_LAYERS);
//
//    save_weak_classifiers(CLASSIFIER_PATH, weak_classifiers);
//    weak_classifiers.clear();
//    weak_classifiers = load_weak_classifiers(CLASSIFIER_PATH);
//
//    size_t i = 0;
//    for (auto &weak_classifier: weak_classifiers) {
//        printf("== Classifier %zu ==\n", i);
//        weak_classifier.feat->print();
//        printf("threshold: %f\n", weak_classifier.threshold);
//        printf("polarity: %d\n", weak_classifier.polarity);
//        printf("alpha: %f\n", weak_classifier.alpha);
//        i++;
//    }

    return 0;
}