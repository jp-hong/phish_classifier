

from time import perf_counter
from os import cpu_count
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import numpy as np
from numba import jit


def print_cv_score(scores, K, label=None, metric=None):
    if label:
        print("Cross validation score: %0.4f (+/- %0.4f) [K=%d / %s / %s]" % \
            (K, scores.mean(), scores.std(), metric, label))
    else:
        print("Cross validation score: %0.4f (+/- %0.4f) [K=%d / %s]" % \
            (K, scores.mean(), scores.std(), metric))


def print_run_time(f_time, p_time, cv_time):
    print("\nElapsed time: %0.2fs" % (f_time + p_time + cv_time), end=" ")
    print("[fit %0.2fs, predict %0.2fs, cv: %0.2fs]" % \
        (f_time, p_time, cv_time))


def workflow(model, label, scoring, K, x_train, x_test, 
    y_train, y_test, n_jobs=4):

    start = perf_counter()
    fit_result = model.fit(x_train, y_train)
    print(fit_result)
    print()
    end = perf_counter()
    f_time = end - start

    start = perf_counter()
    y_pred = model.predict(x_test)

    print("MODEL: {}".format(label))
    print(classification_report(y_test, y_pred))
    end = perf_counter()
    p_time = end - start

    start = perf_counter()

    cv_scores = cross_validate(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring=scoring,
        cv=K,
        n_jobs=n_jobs
    )

    for key in cv_scores.keys():
        print_cv_score(cv_scores[key], K, label=label, metric=key)

    end = perf_counter()
    cv_time = end - start

    print_run_time(f_time, p_time, cv_time)

    return cv_scores


@jit(nopython=True)
def to_bin(y):
    y0 = np.zeros(shape=y.shape, dtype=np.int32)

    for i in range(y.shape[0]):
        if y[i][0] > 0.5:
            y0[i][0] = 1
        else:
            y0[i][1] = 1

    return y0


@jit(nopython=True)
def recall(y_true, y_pred):
    tp, fn = 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i][1] == 1 and y_pred[i][1] == 1:
            tp += 1

        if y_true[i][1] == 1 and y_pred[i][1] == 0:
            fn += 1

    return tp / (tp + fn)


@jit(nopython=True)
def accuracy(y_true, y_pred):
    tp, tn = 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i][1] == 1 and y_pred[i][1] == 1:
            tp += 1

        if y_true[i][1] == 0 and y_pred[i][1] == 0:
            tp += 1

    return (tp + tn) / y_true.shape[0]


@jit(nopython=True)
def to_1D(y):
    l = y.shape[0]
    y0 = np.zeros(l, dtype=np.int32)

    for i in range(l):
        if y[i][1] == 1:
            y0[i] = 1

    return y0


def kfold(model, model_params, x, y, cv=10):
    kf = KFold(n_splits=cv, shuffle=True, random_state=11)

    for train_idx, test_idx in kf.split(x, y):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        model.fit(
            x_train,
            y_train,
            batch_size=model_params["batch_size"],
            epochs=model_params["epochs"],
            verbose=2,
            validation_data=(x_test, y_test),
            callbacks=[model_params["es"]]
        )

        y_pred = model.predict(x_test)

