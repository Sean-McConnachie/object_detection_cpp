#pragma once

#include <vector>
#include "image.h"
#include "feature.h"
#include "utils.h"
#include "learner.h"

typedef struct {
    flt falsePositiveRate;
    flt detectionRate;
} Evaluation;

class AttentionalCascade {
private:
    vec<shdptr<ImgType>> integrals;
    vec<int> labels;
    shdptr<vec<shdptr<Feature>>> features;

    vec<shdptr<ImgType>> validationIntegrals;
    vec<int> validationLabels;

    vec<shdptr<ImgType>> posIntegrals;  // the positive set always stay the same (only contains faces)
    vec<shdptr<ImgType>> negIntegrals;  // the negative set gets reduced on each iteration (only contains non-faces)

    Evaluation evaluate(const classifiervec &weakClassifiers, flt threshold);

    Learner trainStage();

    void reduceFalsePositives(const classifiervec &cascade, flt threshold);
public:
    AttentionalCascade(vec<ImgType> ims,
                       vec<int> lbls,
                       const Features &feats,
                       Stats stats,
                       Samples validation);

    ~AttentionalCascade() = default;

    vec<shdptr<classifiervec>> train(flt maxFalsePositive, flt minDetection, flt targetOverallFalsePositive);

};