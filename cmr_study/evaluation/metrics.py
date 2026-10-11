import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import pearsonr


def classification(y, probabilities, classes):
    y = np.asarray(y)
    prediction = np.argmax(probabilities, axis=1)
    recalls, aucs, missing = {}, [], []
    for c, name in enumerate(classes):
        selected = y == c
        recalls[name] = float(np.mean(prediction[selected] == c)) if selected.any() else None
        if len(np.unique(selected)) < 2:
            aucs.append(None)
            missing.append(name)
        else:
            aucs.append(float(roc_auc_score(selected, probabilities[:, c])))
    return dict(accuracy=float(accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, labels=list(range(len(classes))), average="macro", zero_division=0)),
                macro_auc=float(np.mean(aucs)) if not missing else None,
                auc_unavailable_classes=missing, per_class_auc=dict(zip(classes, aucs)),
                per_class_recall=recalls, confusion=confusion_matrix(y, prediction, labels=range(len(classes))).tolist())


def regression(y, p):
    y, p = np.asarray(y), np.asarray(p)
    valid = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(p) > 0
    difference = p-y
    sd = float(np.std(difference, ddof=1)) if len(y) > 1 else None
    bias = float(np.mean(difference))
    return dict(mae=float(np.mean(np.abs(difference))), rmse=float(np.sqrt(np.mean(difference**2))),
                pearson_r=float(pearsonr(y, p)[0]) if valid else None,
                pearson_unavailable_reason=None if valid else "constant prediction/target or fewer than two patients",
                bias=bias, limits_of_agreement=[bias-1.96*sd, bias+1.96*sd] if sd is not None else None,
                units="EF percentage points")
