

from time import perf_counter
from os import cpu_count
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from .util import print_cv_score


def workflow(model, label, x_train, x_test, y_train, y_test, n_jobs):

    start = perf_counter()
    fit_result = model.fit(x_train, y_train)
    print(fit_result)
    print()
    end = perf_counter()
    fit_time = end - start

    start = perf_counter()
    y_pred = model.predict(x_test)

    print("MODEL: {}".format(label))
    print(classification_report(y_test, y_pred))
    end = perf_counter()
    pred_time = end - start

    start = perf_counter()
    cv_scores = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="accuracy",
        cv=10,
        n_jobs=n_jobs
    )

    print_cv_score(cv_scores, label=label)
    end = perf_counter()
    cv_time = end - start

    print("\nElapsed time: %0.2fs" % (fit_time + pred_time + cv_time), end=" ")
    print("[fit %0.2fs, predict %0.2fs, cv: %0.2fs]" % \
        (fit_time, pred_time, cv_time))

    return cv_scores