from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import *

ys = []
zs = []

with open("o.txt", mode='r') as f:
    for line in f.readlines():
        y, z = line.split()
        ys.append(float(y))
        zs.append(float(z))

ys = np.array(ys)
zs = np.array(zs)


def dist_plot():
    sns.set_style("whitegrid")
    sns.set_palette('muted')

    a = sns.distplot(zs[ys > .5], rug=True, hist=False, color='g', kde_kws={'shade': True})
    sns.distplot(zs[ys < .5], rug=True, hist=False, color='r', kde_kws={'shade': True}, ax=a)

    plt.xlabel('Feature value $z = f(x)$')
    plt.ylabel('Value probability')
    plt.tight_layout()
    plt.savefig('plot.png')


def precision_recall_curver():
    average_precision = average_precision_score(ys, zs)
    precision, recall, thresholds = precision_recall_curve(ys, zs)

    plt.figure(figsize=(18, 5))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    for i, (num_faces, r, t) in enumerate(zip(precision, recall, thresholds)):
        if i % 2 == 0:
            plt.annotate(f'{t:.2f}', xy=(r, num_faces))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.tight_layout()
    plt.savefig('plot.png')


def prediction_statistics(threshold: int):
    PredictionStats = NamedTuple('PredictionStats', [('tn', int), ('fp', int), ('fn', int), ('tp', int)])

    def prediction_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, PredictionStats]:
        c = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = c.ravel()
        return c, PredictionStats(tn=tn, fp=fp, fn=fn, tp=tp)

    theta = threshold
    c, s = prediction_stats(ys, zs >= theta)

    print(f'Precision {s.tp / (s.tp + s.fp):.3}, recall {s.tp / (s.tp + s.fn):.3}.')
    sns.heatmap(c, cmap='YlGnBu', annot=True, square=True,
                xticklabels=['Predicted negative', 'Predicted positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion matrix for $\Theta={theta}$');
    plt.tight_layout()
    plt.savefig('plot.png')


def running_sums():
    num_faces = np.argsort(zs)
    # print(num_faces)
    zss = zs[num_faces]
    # print(zs)
    # print(zss)
    yss = ys[num_faces]
    # print(ys)
    # print(yss)

    s_minuses, s_pluses = [], []
    s_minus, s_plus = 0., 0.
    t_minus, t_plus = 0., 0.

    for z, y in zip(zss, yss):
        if y == 0:
            s_minus += 1
            t_minus += 1
        else:
            s_plus += 1
            t_plus += 1
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)

    errors_1, errors_2 = [], []

    min_e = float('inf')
    min_idx = 0
    polarity = 0
    for i, (s_m, s_p) in enumerate(zip(s_minuses, s_pluses)):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        errors_1.append(error_1)
        errors_2.append(error_2)

        if error_1 < min_e:
            min_e = error_1
            min_idx = i
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            min_idx = i
            polarity = 1

    print(
        f'Minimal error: {min_e:.2} at index {min_idx} with threshold {zs[min_idx]:.2}. Classifier polarity is {polarity}.')

    # plt.figure()
    # plt.plot(zss, errors_1)
    # plt.plot(zss, errors_2)
    # plt.legend(['$S^+ + (T^- - S^-)$', '$S^- + (T^+ - S^+)$', '?'])
    # plt.xlabel('Threshold')
    # plt.ylabel('Errors')
    # plt.tight_layout();

    # plt.figure()
    # plt.plot(zss, s_minuses)
    # plt.plot(zss, s_pluses)
    # plt.legend(['s-', 's+'])
    # plt.xlabel('Threshold')
    # plt.ylabel('Running sum')
    # plt.tight_layout();
    # plt.savefig('plot.png')

    return zss[min_idx]


# dist_plot()
# precision_recall_curver()
thres = running_sums()
prediction_statistics(thres)
# prediction_statistics(0.04)
