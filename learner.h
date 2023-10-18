#pragma once

#include <omp.h>
#include <utility>
#include <fstream>

#include "image.h"
#include "feature.h"
#include "utils.h"

typedef ImgFlt flt;

typedef std::vector<flt> fltvec;
typedef std::vector<int> intvec;

typedef struct {
    flt threshold;
    int polarity;
    flt alpha;
    const shdptr<Feature> feat;

    [[nodiscard]] std::string csv() const;
} WeakClassifier;

typedef std::vector<WeakClassifier> classifiervec;

WeakClassifier
load_weak_classifier(const std::string &csv);

void
save_weak_classifiers(const std::string &path, const std::vector<WeakClassifier> &weakClassifiers);

std::vector<WeakClassifier>
load_weak_classifiers(const std::string &path);

void
print_weak_classifiers(const std::vector<WeakClassifier> &weakClassifiers);

typedef struct {
    flt threshold;
    int polarity;
    flt classification_error;
    shdptr<Feature> feat;
} ClassifierResult;

typedef struct {
    flt threshold;
    int polarity;
} ThresholdPolarity;

typedef struct {
    flt t_minus;
    flt t_plus;
    fltvec s_minuses;
    fltvec s_pluses;
} RunningSums;

typedef struct {
    flt alphaSum;
    flt weightedSum;
    flt confidenceInterval;

    [[nodiscard]]int label() const {
        if (weightedSum >= 0) {
            return 1;
        } else {
            return 0;
        }
    }
} StrongClassifierResult;

class Learner {
private:


    void initWeights();

    void normalizeWeights();

    ClassifierResult applyFeature(shdptr<Feature> feature);

    RunningSums buildRunningSums();

    static int weakClassifier(const ImgType &img, shdptr<Feature> feat, flt threshold, int polarity);

    static int runWeakClassifier(const ImgType &img, const WeakClassifier &weakClassifier_);

    static ThresholdPolarity
    findBestThreshold(const fltvec &results, const RunningSums &runningSums);

    ThresholdPolarity determineThresholdPolarity(const fltvec &results);

public:
    std::vector<shdptr<ImgType>> integrals;
    std::vector<int> labels;
    fltvec weights;
    shdptr<std::vector<shdptr<Feature>>> features;

    shdptr<classifiervec> weakClassifiers;
    std::vector<fltvec> weightHist;

    Learner(std::vector<shdptr<ImgType>> normalizedIntegrals,
            std::vector<int> lbls,
            shdptr<std::vector<shdptr<Feature>>> feats);

    ~Learner() = default;

    void reInit(std::vector<shdptr<ImgType>> integrals, intvec lbls);

    static StrongClassifierResult strongClassifier(const ImgType &img, const classifiervec &weakClassifiers);

    shdptr<classifiervec> train(int numWeakClassifiers);
};
