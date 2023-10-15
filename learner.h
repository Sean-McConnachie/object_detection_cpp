#pragma once

#include <omp.h>

#include "image.h"
#include "feature.h"
#include "utils.h"

typedef ImgFlt flt;

typedef std::vector<flt> fltvec;

typedef struct {
    flt threshold;
    int polarity;
    flt alpha;
    const Feature *feat;
} WeakClassifier;

typedef struct {
    flt threshold;
    int polarity;
    flt classification_error;
    Feature *feat;
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

class Learner {
private:


    void initWeights();

    void normalizeWeights();

    ClassifierResult applyFeature(Feature *feature);

    RunningSums buildRunningSums();

    static int weakClassifier(const ImgType &img, const Feature *feat, flt threshold, int polarity);

    static int runWeakClassifier(const ImgType &img, const WeakClassifier &weakClassifier_);

    static ThresholdPolarity
    findBestThreshold(const fltvec &results, const RunningSums &runningSums);

    ThresholdPolarity determineThresholdPolarity(const fltvec &results);

public:

    std::vector<ImgType> integrals;
    std::vector<int> labels;
    fltvec weights;
    std::vector<const Feature *> features;

    std::vector<WeakClassifier> weakClassifiers;
    std::vector<fltvec> weightHist;

    Learner(std::vector<ImgType> ims, std::vector<int> lbls, const Features &feats, Stats stats);

    ~Learner() = default;

    std::vector<WeakClassifier> train(int numWeakClassifiers);
};
