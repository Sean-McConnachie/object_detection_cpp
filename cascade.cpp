#include "cascade.h"

#include <utility>

AttentionalCascade::AttentionalCascade(vec<ImgType> ims,
                                       vec<int> lbls,
                                       const Features &feats,
                                       Stats stats,
                                       Samples validation) {
    integrals.reserve(ims.size());

    for (int i = 0; i < ims.size(); ++i) {
        auto im = &ims[i];
        im->normalize(stats.mean, stats.std);
        integrals.push_back(mkshd<ImgType>(im->toIntegral()));

        if (lbls[i] == 1) {
            posIntegrals.push_back(integrals[i]);
        } else {
            negIntegrals.push_back(integrals[i]);
        }
    }

    labels = std::move(lbls);
    auto feat_vec = feature_vec(feats);
    features = mkshd<vec<shdptr<Feature>>>(feat_vec);

    for (auto &im: validation.ims) {
        im.normalize(stats.mean, stats.std);
        validationIntegrals.push_back(mkshd<ImgType>(im.toIntegral()));
    }
    validationLabels = std::move(validation.labels);
}

vec<shdptr<classifiervec>>
AttentionalCascade::train(flt maxFalsePositive, flt minDetection, flt targetOverallFalsePositive) {
    flt fPos = maxFalsePositive;                // f
    flt mDec = minDetection;               // d
    flt fPosTar = targetOverallFalsePositive;   // f_target

    fltvec fPosVec = {1.0}; // F
    fltvec mDecVec = {1.0}; // D
    fltvec thresholds = {1.0};

    vec<shdptr<classifiervec>> cascade;

    int i = 0;
    int n;

    auto START_TIME = std::chrono::high_resolution_clock::now();
    while (fPosVec[i] > fPosTar) {
        i++;

        auto stage_start_time = std::chrono::high_resolution_clock::now();
        auto time_since_start = std::chrono::duration_cast<std::chrono::seconds>(
                stage_start_time - START_TIME).count();
        printf("[%lds] CASCADE LAYER %d STARTED\n", time_since_start, i);
        std::cout << std::endl;

        n = 0;
        fPosVec.push_back(fPosVec[i - 1]);
        mDecVec.push_back(mDecVec[i - 1]);
        thresholds.push_back(1.0);
        Learner learner = trainStage();
        shdptr<classifiervec> classifiers = learner.train(n);
        while (fPosVec[i] > fPos * fPosVec[i - 1]) {
            n++;
            classifiers = learner.train(n);
            auto eval = evaluate(*classifiers, thresholds[i]);
            fPosVec[i] = eval.falsePositiveRate;
            mDecVec[i] = eval.detectionRate;

            while (mDecVec[i] < mDec * mDecVec[i - 1]) {
                const flt DECREMENT_AMT = 0.01;
                thresholds[i] -= DECREMENT_AMT;

                eval = evaluate(*classifiers, thresholds[i]);
                fPosVec[i] = eval.falsePositiveRate;
                mDecVec[i] = eval.detectionRate;
            }
            std::cout << std::endl;
            printf("==== fPosVec[%d]: %f, thresholds[%d]: %f\n", i, fPosVec[i], i, thresholds[i]);
        }

        auto time_since_stage_start = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - stage_start_time).count();
        printf("CASCADE LAYER %d FINISHED [%lds] === fPos[%d]: %f, mDec[%d]: %f, thresholds[i]: %f | TARGET_DIFF = %f\n",
               i, time_since_stage_start, i, fPosVec[i], i, mDecVec[i], thresholds[i], fPosTar - fPosVec[i]);

        if (fPosVec[i] > fPosTar) {
            reduceFalsePositives(*classifiers, thresholds[i]);
        }
        printf("Number of negative samples left: %zu\n", negIntegrals.size());

        cascade.push_back(classifiers);
    }
    return cascade;
}

void
AttentionalCascade::reduceFalsePositives(const classifiervec &cascade, flt threshold) {
    size_t n = negIntegrals.size();
    for (int i = 0; i < n; ++i) {
        auto img = negIntegrals[i];
        auto h = Learner::strongClassifier(*img, cascade);
        auto is_rejected = h.confidenceInterval < threshold;
        if (is_rejected) continue;
        if (h.label() == 0) {
            negIntegrals.erase(negIntegrals.begin() + i);
            i--;
            n--;
        }
    }
}

Learner
AttentionalCascade::trainStage() {
    vec<shdptr<ImgType>> ims;
    ims.reserve(posIntegrals.size() + negIntegrals.size());
    ims.insert(ims.end(), posIntegrals.begin(), posIntegrals.end());
    ims.insert(ims.end(), negIntegrals.begin(), negIntegrals.end());

    vec<int> lbls;
    lbls.reserve(posIntegrals.size() + negIntegrals.size());
    lbls.insert(lbls.end(), posIntegrals.size(), 1);
    lbls.insert(lbls.end(), negIntegrals.size(), 0);

    return {ims, lbls, features};
}

Evaluation
AttentionalCascade::evaluate(const classifiervec &weakClassifiers, flt threshold) {
    size_t n = validationLabels.size();
    int falsePositive = 0;
    int truePositive = 0;
    int truePositives = 0;

    for (int i = 0; i < n; ++i) {
        auto img = validationIntegrals[i];
        auto label = validationLabels[i];
        auto h = Learner::strongClassifier(*img, weakClassifiers);
        auto is_rejected = h.confidenceInterval < threshold;
        if (is_rejected) continue;

        auto lbl = h.label();
        if (lbl == 1 && label == 1) { truePositive++; }
        if (lbl == 1 && label == 0) { falsePositive++; }

        if (label == 1) { truePositives++; }
    }
    Evaluation eval{};
    eval.falsePositiveRate = falsePositive / (flt) (n);
    eval.detectionRate = truePositive / (flt) (truePositives);
    return eval;
}
