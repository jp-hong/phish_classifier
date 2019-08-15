

from time import perf_counter
from os import cpu_count
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def print_cv_score(scores, label=None, metric=None):
    if label:
        print("Cross validation score: %0.4f (+/- %0.4f) [%s / %s]" % \
            (scores.mean(), scores.std(), metric, label))
    else:
        print("Cross validation score: %0.4f (+/- %0.4f) [%s]" % \
            (scores.mean(), scores.std(), metric))


def print_run_time(f_time, p_time, cv_time):
    print("\nElapsed time: %0.2fs" % (f_time + p_time + cv_time), end=" ")
    print("[fit %0.2fs, predict %0.2fs, cv: %0.2fs]" % \
        (f_time, p_time, cv_time))


def workflow(model, label, metric, x_train, x_test, y_train, y_test, n_jobs):

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
    cv_scores = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring=metric,
        cv=10,
        n_jobs=n_jobs
    )

    print_cv_score(cv_scores, label=label, metric=metric)
    end = perf_counter()
    cv_time = end - start

    print_run_time(f_time, p_time, cv_time)

    return cv_scores