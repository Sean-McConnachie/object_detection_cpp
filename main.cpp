#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "constants.h"
#include "utils.h"
#include "feature.h"
#include "learner.h"
#include "cascade.h"
#include "runtime.h"

int train_manual(int numClassifiers) {
    const int FACE_COUNT = 1000;
    const int BG_COUNT = 1000;

    const char *CLASSIFIER_DIR = "../classifiers/paper_impl";

    if (mkdir(CLASSIFIER_DIR, 0777) == -1) {
    } else {
        printf("Created cascade directory: %s\n", CLASSIFIER_DIR);
    }

    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    auto samples = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
    auto stats = compute_stats(samples.ims);
    printf("mean: %f, std: %f\n", stats.mean, stats.std);

    auto features = generate_features();
    print_features(features);
    vec<shdptr<Feature>> fvec = feature_vec(features);

    vec<shdptr<ImgType>> integrals;
    for (auto &im: samples.ims) {
        im.normalize(stats.mean, stats.std);
        auto integral = im.toIntegral();
        integrals.push_back(std::make_shared<ImgType>(integral));
    }

    Learner learner(integrals, samples.labels, mkshd(fvec));
    learner.train(numClassifiers);

    // save to file
    char path[300];
    sprintf(path, "%s/%d.csv", CLASSIFIER_DIR, numClassifiers);
    save_weak_classifiers(path, *learner.weakClassifiers);

    return 0;
}

int train_cascade() {
    const int FACE_COUNT = 2500;
    const int BG_COUNT = 2500;
    const double MAX_FALSE_POSITIVE = 0.005;
    const double MIN_DETECTION = 0.995;
    const double TARGET_OVERALL_FALSE_POSITIVE = 0.0025;

    const char *CLASSIFIER_DIR = "../classifiers/";

    char dir[300];
    sprintf(dir, "%s%dfcs_%dbgs_%dpx_%fFP_%fD_%fTFP",
            CLASSIFIER_DIR, FACE_COUNT, BG_COUNT, FEATURE_SIZE, MAX_FALSE_POSITIVE, MIN_DETECTION,
            TARGET_OVERALL_FALSE_POSITIVE
    );
    if (mkdir(dir, 0777) == -1) {
        printf("Cascade already exists!");
        return 1;
    } else {
        printf("Created cascade directory: %s\n", dir);
    }

    auto face_paths = list_dir(FP_FACES_DIR);
    auto bg_paths = list_dir(FP_BGS_DIR);
    auto samples = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
    auto validation = sample_data(FACE_COUNT, BG_COUNT, face_paths, bg_paths);
    auto stats = compute_stats(samples.ims);
    printf("mean: %f, std: %f\n", stats.mean, stats.std);

    auto features = generate_features();
    print_features(features);

    auto cascade = AttentionalCascade(samples.ims, samples.labels, features, stats, validation);
    auto cascade_classifiers = cascade.train(MAX_FALSE_POSITIVE, MIN_DETECTION, TARGET_OVERALL_FALSE_POSITIVE);

    for (int i = 0; i < cascade_classifiers.size(); ++i) {
        auto c = cascade_classifiers[i];
        char path[300];
        sprintf(path, "%s/%d.csv", dir, i);
        save_weak_classifiers(path, *c);
    }

    return 0;
}

int test_image() {
    const char *CLASSIFIER_DIR = "../classifiers/paper_impl";
    const char *IMAGE_PATH = "../dataset/solvay-conference.jpg";

    vec<classifiervec> cascade;
    {
        std::vector<std::string> files;
        for (const auto &entry: std::filesystem::directory_iterator(CLASSIFIER_DIR)) {
            files.push_back(entry.path());
        }
        for (const auto &file: files) {
            cascade.push_back(load_weak_classifiers(file));
        }
        std::sort(cascade.begin(), cascade.end(), [](const classifiervec &a, const classifiervec &b) {
            return a.size() < b.size();
        });
    }

    ImgType integral(0, 0);
    cv::Mat cvim;
    {
        cv::Mat tmp = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
        Scale scaled = scaled_size({IM_WIDTH, IM_HEIGHT}, {tmp.cols, tmp.rows});
        cv::resize(tmp, cvim, {scaled.width, scaled.height});

        integral = ImgType(cvim.rows, cvim.cols);
        integral.loadGrayScale(cvim);
//        integral.rangeTo(255.0);
        integral.normalize();
        integral = integral.toIntegral();
    }

    auto runtime = Runtime(cascade);
    auto boxes = runtime.run(integral);
    Runtime::drawBoxes(cvim, boxes);

    cv::imshow("image", cvim);
    cv::waitKey(0);

    return 0;
}

enum Mode {
    TrainManual,
    TrainCascade,
    TestImage
};

int main() {
//    intvec intervals = {26, 50,51, 52, 100};
//    for (int interval: intervals) {
//        printf("interval: %d\n", interval);
//        train_manual(interval);
//    }
//
//    return 0;
    switch (TestImage) {
        case TrainManual:
            return train_manual(0);
        case TrainCascade:
            return train_cascade();
        case TestImage:
            return test_image();
    }
}