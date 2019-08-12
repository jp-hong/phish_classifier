

from os import cpu_count
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from .util import print_cv_score


def workflow(model, label, x_train, x_test, y_train, y_test):

    fit_result = model.fit(x_train, y_train)
    print(fit_result)
    print()

    y_pred = model.predict(x_test)

    print("MODEL: {}".format(label))
    print(classification_report(y_test, y_pred))

    cv_scores = cross_val_score(
        estimator=model,
        X=x_train,
        y=y_train,
        scoring="accuracy",
        cv=10,
        n_jobs=cpu_count()-1
    )

    print_cv_score(cv_scores, label=label)

    return cv_scores