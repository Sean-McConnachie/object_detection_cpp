#include "learner.h"

#include <utility>


std::string
WeakClassifier::csv() const {
    std::string out = std::to_string(this->threshold);
    out += ",";
    out += std::to_string(this->polarity);
    out += ",";
    out += std::to_string(this->alpha);
    out += ",";
    out += this->feat->csv();
    return out;
}

WeakClassifier
load_weak_classifier(const std::string &csv) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream in(csv);
    while (std::getline(in, part, ',')) {
        parts.push_back(part);
    }

    flt threshold = std::stof(parts[0]);
    int polarity = std::stoi(parts[1]);
    flt alpha = std::stof(parts[2]);
    std::string feat_name = parts[3];
    size_t x = std::stoi(parts[4]);
    size_t y = std::stoi(parts[5]);
    size_t w = std::stoi(parts[6]);
    size_t h = std::stoi(parts[7]);

    shdptr<Feature> feat;
    if (feat_name == "2h") {
        feat = std::make_shared<Feature2h>(x, y, w, h);
    } else if (feat_name == "2v") {
        feat = std::make_shared<Feature2v>(x, y, w, h);
    } else if (feat_name == "3h") {
        feat = std::make_shared<Feature3h>(x, y, w, h);
    } else if (feat_name == "3v") {
        feat = std::make_shared<Feature3v>(x, y, w, h);
    } else if (feat_name == "4r") {
        feat = std::make_shared<Feature4>(x, y, w, h);
    } else {
        throw std::runtime_error("Unknown feature name: " + feat_name);
    }

    return {threshold, polarity, alpha, feat};
}

void
save_weak_classifiers(const std::string &path, const std::vector<WeakClassifier> &weakClassifiers) {
    std::ofstream out(path);
    for (const auto &c: weakClassifiers) {
        out << c.csv() << std::endl;
    }
    out.close();
}

std::vector<WeakClassifier>
load_weak_classifiers(const std::string &path) {
    std::vector<WeakClassifier> weakClassifiers;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        weakClassifiers.push_back(load_weak_classifier(line));
    }
    in.close();
    return weakClassifiers;
}

void
print_weak_classifiers(const std::vector<WeakClassifier> &weakClassifiers) {
    int i = 0;
    for (const auto &c: weakClassifiers) {
        printf("%d Feature%s(%d, %zu, %zu,%zud) threshold: %f polarity: %d alpha: %f\n", i++, c.feat->name(), c.feat->x,
               c.feat->y, c.feat->width, c.feat->height, c.threshold, c.polarity, c.alpha);
    }
}

Learner::Learner(std::vector<shdptr<ImgType>> normalizedIntegrals,
                 std::vector<int> lbls,
                 shdptr<std::vector<shdptr<Feature>>> feats) {
    integrals = std::move(normalizedIntegrals);

    labels = std::move(lbls);

    initWeights();

    features = std::move(feats);

    weakClassifiers = std::make_shared<classifiervec>();
}

void
Learner::initWeights() {
    int num_faces, num_bgs;
    num_bgs = num_faces = 0;
    for (const int y: labels) {
        if (y == 1) num_faces++;
        else num_bgs++;
    }

    weights.reserve(labels.size());
    flt w_face = 1.0 / (2.0 * num_faces);
    flt w_bg = 1.0 / (2.0 * num_bgs);
    for (const int y: labels) {
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
Learner::weakClassifier(const ImgType &img, shdptr<Feature> feat, flt threshold, int polarity) {
    auto r = feat->diff(img);
    if ((flt) polarity * r < (flt) polarity * threshold) {
        return 1;
    } else {
        return 0;
    }
}

int
inline
Learner::runWeakClassifier(const ImgType &img, const WeakClassifier &weakClassifier_) {
    return weakClassifier(img, weakClassifier_.feat, weakClassifier_.threshold, weakClassifier_.polarity);
}

StrongClassifierResult
Learner::strongClassifier(const ImgType &img, const classifiervec &weakClassifiers) {
    flt sum_hypotheses = 0;
    flt sum_alphas = 0;
    for (const auto &c: weakClassifiers) {
        sum_hypotheses += c.alpha * (flt) runWeakClassifier(img, c);
        sum_alphas += c.alpha;
    }
    return {sum_alphas, sum_hypotheses, std::abs(sum_hypotheses / sum_alphas)};
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
Learner::applyFeature(shdptr<Feature> feature) {
    fltvec results;
    std::fill_n(std::back_inserter(results), integrals.size(), 0);

#pragma omp parallel for
    for (int i = 0; i < integrals.size(); ++i) {
        results[i] = feature->diff(*integrals[i]);
    }

    ThresholdPolarity result = determineThresholdPolarity(results);

    flt classification_error = 0.0;
    for (int i = 0; i < integrals.size(); ++i) {
        auto im = integrals[i];
        auto label = labels[i];
        auto weight = weights[i];

        auto h = weakClassifier(*im, feature, result.threshold, result.polarity);
        classification_error += weight * std::abs(h - label);
    }

    return {result.threshold, result.polarity, classification_error, feature};
}

shdptr<classifiervec>
Learner::train(int numWeakClassifiers) {
    const size_t TOTAL_CLASSIFIERS = numWeakClassifiers * features->size();
    size_t run_classifiers = 0;

    auto total_start = std::chrono::high_resolution_clock::now();
    for (int t = (int) weakClassifiers->size(); t < numWeakClassifiers; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        normalizeWeights();

        int status = STATUS_EVERY;

        ClassifierResult best{0, 0, std::numeric_limits<flt>::max(), nullptr};

        size_t i = 0;
        for (const auto &feat: *features) {
            --status;
            ++run_classifiers;

            bool improved = false;

            ClassifierResult result = applyFeature(feat);
            if (result.classification_error < best.classification_error) {
                improved = true;
                best = result;
            }

            if (improved || status == 0) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
                        end - total_start).count();
                auto features_perc = 100.0 * (flt) i / (flt) features->size();
                printf("\t[%lds]\t(%d/%d)\t| Stage: [%ldms] %.2f%% (%zu/%zu)",
                       total_duration, t + 1, numWeakClassifiers, duration, features_perc, i + 1, features->size());
                if (improved)
                    printf(" \tError improved to %f\t%s", best.classification_error, best.feat->str().c_str());

                auto remaining_time =
                        ((flt) total_duration / (flt) run_classifiers) * ((flt) (TOTAL_CLASSIFIERS - run_classifiers));
                printf(" \tRemaining time: %lds", (long) remaining_time);

                std::cout << std::endl;

                status = STATUS_EVERY;
            }
            ++i;
        }
        auto beta = best.classification_error / (1.0 - best.classification_error);
        auto alpha = std::log(1.0 / beta);

        WeakClassifier classifier{best.threshold, best.polarity, (flt) alpha, best.feat};

        for (i = 0; i < integrals.size(); ++i) {
            auto im = integrals[i];
            auto label = labels[i];
            auto h = runWeakClassifier(*im, classifier);
            auto e = std::abs(h - label);
            weights[i] = weights[i] * std::pow(beta, 1 - e);
        }

        weakClassifiers->push_back(classifier);
    }
    return weakClassifiers;
}