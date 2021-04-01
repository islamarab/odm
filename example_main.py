""" Main module """


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sm

from utils.metrics import preprocess_df, remove_overlapping_objects, calculate_metrics
from utils.plots import plot_confusion_matrix


def main():
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 7

    # Classes
    classes = ["dog", "cat", "Null"]
    # classes = ["dog", "cat"]

    # DataFrames
    actual_df = pd.read_csv("example\\actual.csv")
    actual_df = preprocess_df(actual_df)

    detected_df = pd.read_csv("example\\detected.csv")
    detected_df = preprocess_df(detected_df)
    detected_df = remove_overlapping_objects(detected_df)

    # Calculating
    df = calculate_metrics(actual_df, detected_df, prob_thresh=0, iou_thresh=0.0)

    df.to_csv("example\\result_df.csv", index=False)

    # ============ Collect data for sklearn =============
    y_true = []
    y_pred = []
    y_score = []
    for i, row in df[df['a_xmin'] != 'Null'].iterrows():

        true_class = row['a_label']
        y_true.append(true_class)
        pred_class = row['d_label']
        y_pred.append(pred_class)

        prob = row['d_prob']
        if prob == "Null":
            y_score.append(0)
        else:
            y_score.append(float(prob))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # for true, pred in zip(y_true, y_pred):
    #     print(true, pred)

    print("Accuracy ", 100 * (y_true == y_pred).sum() / len(y_true))

    # ========= Confusion Matrix ===========
    cm = sm.confusion_matrix(y_true, y_pred, labels=sorted(classes))
    plot_confusion_matrix(cm, classes=sorted(classes))
    plt.show()

    cm_display = sm.ConfusionMatrixDisplay(cm, display_labels=sorted(classes)).plot()
    plt.show()

    # ========= Classification Report ===========

    cp = sm.classification_report(y_true, y_pred, labels=sorted(classes), output_dict=False)
    print(cp)

    # ========= PR Curve ===========
    precision = {}
    recall = {}
    thresh = {}

    for i in classes:
        precision[i], recall[i], thresh[i] = sm.precision_recall_curve(y_true, y_score, pos_label=i)
        plt.plot(recall[i], precision[i], lw=2, label=f'{i}')

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()

    print("PR Curve")

    # for pr, rec, thresh_ in zip(precision["full_lined"], recall["full_lined"], thresh["full_lined"]):
    #     print(pr, rec, thresh_)

    # ========= ROC Curve ===========
    fpr = {}
    tpr = {}
    thresh = {}
    roc_auc = {}

    for i in classes:
        fpr[i], tpr[i], thresh[i] = sm.roc_curve(y_true, y_score, pos_label=i)
        roc_auc[i] = sm.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{i} (area = {roc_auc[i]:0.2f})')

    ns_probs = [0 for _ in range(len(y_true))]
    ns_fpr, ns_tpr, _ = sm.roc_curve(y_true, ns_probs, pos_label="nolines")
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()

    print("ROC Curve")

    # for tpr_, fpr_, thresh_ in zip(tpr["full_lined"], fpr["full_lined"], thresh["full_lined"]):
    #     print(tpr_, fpr_, thresh_)


if __name__ == "__main__":
    main()
