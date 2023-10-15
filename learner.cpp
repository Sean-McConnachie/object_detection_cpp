#include "learner.h"

#include <utility>

Learner::Learner(std::vector<ImgType> ims, std::vector<int> lbls, const Features &feats, Stats stats) {
    integrals.reserve(ims.size());
    for (auto &im: ims) {
        im.normalize((ImgFlt) stats.mean, (ImgFlt) stats.std);
        auto integral = im.toIntegral();
        integrals.push_back(integral);
    }

    labels = std::move(lbls);

    initWeights();

    features = feature_vec(feats);
}

void
Learner::initWeights() {
    int num_faces, num_bgs;
    num_bgs = num_faces = 0;
    for (const auto &y: labels) {
        if (y == 1) num_faces++;
        else num_bgs++;
    }

    weights.reserve(labels.size());
    flt w_face = 1.0 / (2.0 * num_faces);
    flt w_bg = 1.0 / (2.0 * num_bgs);
    for (const auto &y: labels) {
        if (y == 1) weights.push_back(w_face);
        else weights.push_back(w_bg);
    }
}

void
Learner::normalizeWeights() {
    flt sum = 0;
    for (const auto &w: weights) {
        sum += w;
    }
    for (auto &w: weights) {
        w /= sum;
    }
}

int
Learner::weakClassifier(const ImgType &img, const Feature *feat, flt threshold, int polarity) {
    auto r = feat->diff(img);
    if ((flt) polarity * r < (flt) polarity * threshold) {
        return 1;
    } else {
        return 0;
    }
}

int
Learner::runWeakClassifier(const ImgType &img, const WeakClassifier &weakClassifier_) {
    return weakClassifier(img, weakClassifier_.feat, weakClassifier_.threshold, weakClassifier_.polarity);
}

RunningSums
Learner::buildRunningSums() {
    flt s_minus, s_plus, t_minus, t_plus;
    s_minus = s_plus = t_minus = t_plus = 0.0;
    fltvec s_minuses, s_pluses;
    s_minuses.reserve(integrals.size());
    s_pluses.reserve(integrals.size());

    for (int i = 0; i < integrals.size(); ++i) {
        auto label = labels[i];
        auto weight = weights[i];

        if (label == 0) {
            s_minus += weight;
            t_minus += weight;
        } else {
            s_plus += weight;
            t_plus += weight;
        }
        s_minuses.push_back(s_minus);
        s_pluses.push_back(s_plus);
    }
    return {t_minus, t_plus, s_minuses, s_pluses};
}

ThresholdPolarity
Learner::findBestThreshold(const fltvec &results, const RunningSums &runningSums) {
    flt min_error = std::numeric_limits<flt>::max();
    flt min_z = 0;
    int polarity = 0;

    for (int i = 0; i < results.size(); i++) {
        flt result = results[i];
        flt s_m = runningSums.s_minuses[i];
        flt s_p = runningSums.s_pluses[i];
        flt err1 = s_p + (runningSums.t_minus - s_m);
        flt err2 = s_m + (runningSums.t_plus - s_p);
        if (err1 < min_error) {
            min_error = err1;
            min_z = result;
            polarity = -1;
        } else {
            min_error = err2;
            min_z = result;
            polarity = 1;
        }
    }
    return {min_z, polarity};
}

ThresholdPolarity
Learner::determineThresholdPolarity(const fltvec &results) {
    std::vector<int> sorted(results.size());
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [&results](int i1, int i2) { return results[i1] < results[i2]; });

    for (int i = 0; i < sorted.size(); ++i) {
        if (sorted[i] == i) continue;
        std::swap(labels[i], labels[sorted[i]]);
        std::swap(weights[i], weights[sorted[i]]);
        integrals[i].swap(integrals[sorted[i]]);
    }

    RunningSums sums = buildRunningSums();

    ThresholdPolarity best_threshold = findBestThreshold(results, sums);
    return best_threshold;
}

ClassifierResult
Learner::applyFeature(Feature *feature) {
    fltvec results;
    std::fill_n(std::back_inserter(results), integrals.size(), 0);

    #pragma omp parallel for
    for (int i = 0; i < integrals.size(); ++i) {
        results[i] = feature->diff(integrals[i]);
    }

    ThresholdPolarity result = determineThresholdPolarity(results);

    flt classification_error = 0.0;
    for (int i = 0; i < integrals.size(); ++i) {
        auto im = &integrals[i];
        auto label = labels[i];
        auto weight = weights[i];

        auto h = weakClassifier(*im, feature, result.threshold, result.polarity);
        classification_error += weight * std::abs(h - label);
    }

    return {result.threshold, result.polarity, classification_error, feature};
}

std::vector<WeakClassifier>
Learner::train(int numWeakClassifiers) {
    auto temp = Feature2v(2, 3, 10, 4);

    auto total_start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < numWeakClassifiers; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        normalizeWeights();

        int status = STATUS_EVERY;

        ClassifierResult best{0, 0, std::numeric_limits<flt>::max(), nullptr};

        size_t i = 0;
        for (const auto &feat: features) {
            auto f = const_cast<Feature *>(feat);
//            auto f = &temp;
            --status;
            bool improved = false;

            ClassifierResult result = applyFeature(f);
            if (result.classification_error < best.classification_error) {
                improved = true;
                best = result;
            }

            if (improved || status == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                auto total_duration = std::chrono::duration_cast<std::chrono::seconds >(
                        end - total_start).count();

                if (improved) {
                    printf("t=%d/%d %lds (%ldms in this stage) %zu/%zu %.2f%% evaluated. Classification error improved to %f using %s ...\n",
                           t + 1, numWeakClassifiers, total_duration, duration, i + 1, features.size(),
                           100.0 * i / features.size(), best.classification_error, best.feat->str().c_str());
                } else {
                    printf("t=%d/%d %lds (%ldms in this stage) %zu/%zu %.2f%% evaluated.\n",
                           t + 1, numWeakClassifiers, total_duration, duration, i + 1, features.size(),
                           100.0 * i / features.size());
                }

                status = STATUS_EVERY;
            }
            ++i;
        }
        auto beta = best.classification_error / (1.0 - best.classification_error);
        auto alpha = std::log(1.0 / beta);

        WeakClassifier classifier{best.threshold, best.polarity, alpha, best.feat};

        for (i = 0; i < integrals.size(); ++i) {
            auto im = &integrals[i];
            auto label = labels[i];
            auto h = runWeakClassifier(*im, classifier);
            auto e = std::abs(h - label);
            weights[i] = weights[i] * std::pow(beta, 1 - e);
        }

        weakClassifiers.push_back(classifier);
        weightHist.push_back(weights);
    }
    return weakClassifiers;
}